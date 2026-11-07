// Private stdio IPC for rose.app (--brain claude). Drop-in for codex-voice-bridge.mjs:
// same ops (open, submit, interrupt, close, tool_result) and events (started, delta,
// message, done, tool, tool_cancelled, failure), so CodexConductor runs unchanged.
//
// The brain is Claude through the Claude Agent SDK, which drives the local Claude Code
// CLI with its subscription login. ANTHROPIC_API_KEY is removed from the child env and
// the session is refused unless Claude Code reports subscription (OAuth) auth, so this
// never falls back to pay-per-token API billing.
//
// Models: the conversation session uses ROSE_CLAUDE_TALK_MODEL (default haiku, fastest
// to first word); app sessions use ROSE_CLAUDE_TASK_MODEL (default sonnet).
import {createInterface} from 'node:readline';
import {randomUUID} from 'node:crypto';
import {query, tool, createSdkMcpServer} from '@anthropic-ai/claude-agent-sdk';
import * as z from 'zod/v4';

const TALK_MODEL = process.env.ROSE_CLAUDE_TALK_MODEL || 'haiku';
const TASK_MODEL = process.env.ROSE_CLAUDE_TASK_MODEL || 'sonnet';
const TOOL_TIMEOUT_MS = 60000;
// Everything the conversation session writes is spoken by a TTS voice.
const VOICE_STYLE = `
Your words are spoken aloud by a text-to-speech voice, live. Write the way a person talks:
no markdown, bullets, numbered lists, headings, emoji, URLs or code. Usually one to three short sentences.
When you start a phone task, say briefly that you are on it; the result is announced separately when it finishes.`;
const SUBSCRIPTION_AUTH = new Set(['oauth', '/login managed key', 'none']);

const sessions = new Map(), calls = new Map();
let nextCall = 0;
const send = value => process.stdout.write(JSON.stringify(value) + '\n');
const log = (...a) => process.stderr.write('[claude-bridge] ' + a.join(' ') + '\n');
const fail = () => ({success: false, contentItems: [{type: 'inputText', text: 'Tool unavailable or cancelled.'}]});
const cancelled = (session, callId) => send({session, event: 'tool_cancelled', callId});

function childEnv() {
  const env = {...process.env, CLAUDE_AGENT_SDK_CLIENT_APP: 'rose-voice/1'};
  delete env.ANTHROPIC_API_KEY;      // never bill the API
  delete env.ANTHROPIC_AUTH_TOKEN;
  return env;
}

// Codex-style tool result -> MCP CallToolResult.
function toMcp(result) {
  const content = [];
  for (const item of result?.contentItems || []) {
    if (item.type === 'inputText') content.push({type: 'text', text: item.text});
    else if (item.type === 'inputImage') {
      const m = /^data:(image\/[a-z]+);base64,(.*)$/s.exec(item.imageUrl || '');
      if (m) content.push({type: 'image', mimeType: m[1], data: m[2]});
    }
  }
  if (!content.length) content.push({type: 'text', text: '{}'});
  return {content, isError: result?.success === false};
}

function shapeOf(schema) {
  try {
    const s = z.fromJSONSchema(schema && schema.type ? schema : {type: 'object', properties: {}});
    return s.shape || {};
  } catch (e) {
    log('schema conversion failed:', e.message);
    return {};
  }
}

class Session {
  constructor(id, m) {
    this.id = id;
    this.thread = randomUUID();
    this.model = id === 'conversation' ? TALK_MODEL : TASK_MODEL;
    this.inputs = [];          // user messages waiting for the SDK to pull
    this.wake = null;
    this.pendingTurns = [];    // submitted, not yet sent to the model
    this.active = null;        // {id, interrupted, text}
    this.closed = false;
    const server = createSdkMcpServer({
      name: 'rose', alwaysLoad: true, timeout: TOOL_TIMEOUT_MS + 5000,
      tools: m.tools.map(t => tool(t.name, t.description || t.name, shapeOf(t.inputSchema), args => this.toolCall(t.name, args))),
    });
    const self = this;
    async function* prompt() {
      while (!self.closed) {
        if (self.inputs.length) { yield self.inputs.shift(); continue; }
        await new Promise(r => { self.wake = r; });
        self.wake = null;
      }
    }
    this.q = query({prompt: prompt(), options: {
      model: this.model,
      systemPrompt: id === 'conversation' ? m.instructions + VOICE_STYLE : m.instructions,
      cwd: m.cwd,
      tools: [],                         // no Bash/Read/Edit/web: only Rose's tools
      mcpServers: {rose: server},
      strictMcpConfig: true,
      settingSources: [],                // no CLAUDE.md, user hooks or plugins
      allowedTools: m.tools.map(t => `mcp__rose__${t.name}`),
      includePartialMessages: true,
      persistSession: false,
      thinking: {type: 'disabled'},
      env: childEnv(),
      stderr: d => log(id, String(d).trim()),
    }});
  }

  publish(event) { if (sessions.get(this.id) === this) send({session: this.id, threadId: this.thread, ...event}); }

  // Refuse anything but a subscription login, then start reading the stream.
  async init() {
    const acct = await this.q.accountInfo();
    const src = acct?.apiKeySource;
    log(this.id, 'model', this.model, 'account', JSON.stringify({sub: acct?.subscriptionType, token: acct?.tokenSource, key: src, provider: acct?.apiProvider}));
    // A subscription shows as a plan name (interactive login) or an OAuth token from
    // `claude setup-token` (CLAUDE_CODE_OAUTH_TOKEN, used under launchd). API keys never pass.
    const oauth = Boolean(acct?.subscriptionType) || /OAUTH|claude\.ai/i.test(acct?.tokenSource || '');
    if (!oauth || (src && !SUBSCRIPTION_AUTH.has(src)) || (acct.apiProvider && acct.apiProvider !== 'firstParty')) {
      const err = Error('Claude Code is not signed in with a subscription');
      err.code = 'chatgpt_login_required';
      throw err;
    }
    this.reader = this.read();
    return {threadId: this.thread, serviceTier: this.model, authMode: 'claude-' + (acct.subscriptionType || 'oauth-token')};
  }

  toolCall(name, args) {
    const turnId = this.active?.id;
    return new Promise(resolve => {
      const callId = String(++nextCall);
      const timer = setTimeout(() => { calls.delete(callId); resolve(toMcp(fail())); cancelled(this.id, callId); }, TOOL_TIMEOUT_MS);
      calls.set(callId, {resolve: r => resolve(toMcp(r)), timer, session: this.id});
      this.publish({event: 'tool', callId, params: {tool: name, arguments: args, callId, turnId, threadId: this.thread}});
    });
  }

  submit(text) {
    const id = randomUUID();
    this.pendingTurns.push({id, text});
    this.pump();
    return {turn: {id}};
  }

  // One user message in flight at a time, so each SDK result maps to exactly one turn.
  pump() {
    if (this.active || !this.pendingTurns.length || this.closed) return;
    const t = this.pendingTurns.shift();
    this.active = {id: t.id, interrupted: false, text: ''};
    this.publish({event: 'started', turnId: t.id});
    this.inputs.push({type: 'user', message: {role: 'user', content: t.text}, parent_tool_use_id: null,
                      origin: {kind: 'human'}});
    this.wake?.();
  }

  async interrupt() {
    for (const [id, call] of calls) if (call.session === this.id) {
      clearTimeout(call.timer); call.resolve(fail()); calls.delete(id); cancelled(this.id, id);
    }
    this.pendingTurns = [];
    if (!this.active) return {};
    this.active.interrupted = true;
    try { await this.q.interrupt(); } catch (e) { log(this.id, 'interrupt failed:', e.message); }
    return {};
  }

  finishTurn(status) {
    const t = this.active;
    if (!t) return;
    this.active = null;
    this.publish({event: 'done', turn: {id: t.id, status: t.interrupted ? 'interrupted' : status, error: null}});
    this.pump();
  }

  async read() {
    try {
      for await (const msg of this.q) {
        if (!this.active) continue;
        const turnId = this.active.id;
        if (msg.type === 'stream_event') {
          const ev = msg.event;
          if (ev?.type === 'content_block_delta' && ev.delta?.type === 'text_delta' && !msg.parent_tool_use_id) {
            this.active.text += ev.delta.text;
            this.publish({event: 'delta', turnId, delta: ev.delta.text});
          }
        } else if (msg.type === 'assistant' && !msg.parent_tool_use_id) {
          const blocks = msg.message?.content || [];
          const text = blocks.filter(b => b.type === 'text').map(b => b.text).join('');
          const usesTool = blocks.some(b => b.type === 'tool_use');
          if (text) this.publish({event: 'message', text, phase: usesTool ? 'commentary' : 'final_answer', turnId});
          if (text && usesTool) {
            // Text before a tool call was already spoken via deltas; separate it from what follows.
            this.publish({event: 'delta', turnId, delta: ' '});
          }
        } else if (msg.type === 'result') {
          this.finishTurn(msg.subtype === 'success' && !msg.is_error ? 'completed' : 'failed');
        }
      }
    } catch (e) {
      log(this.id, 'stream error:', e.message);
      if (this.active) this.finishTurn('failed');
      this.publish({event: 'failure', message: 'Claude process stopped'});
    }
  }

  close() {
    this.closed = true;
    this.wake?.();
    try { this.q.close?.(); } catch {}
  }
}

async function handle(m) {
  if (m.op === 'tool_result') {
    const call = calls.get(m.callId);
    if (call) { calls.delete(m.callId); clearTimeout(call.timer); call.resolve(m.result); }
    return;
  }
  if (m.op === 'open') {
    if (sessions.has(m.session)) throw Error('Session already exists');
    const s = new Session(m.session, m);
    sessions.set(m.session, s);
    try {
      const info = await s.init();
      return {...info, model: m.model};
    } catch (e) { s.close(); sessions.delete(m.session); throw e; }
  }
  const s = sessions.get(m.session);
  if (!s) throw Error('Unknown session');
  if (m.op === 'submit') return s.submit(m.text);
  if (m.op === 'interrupt') return s.interrupt();
  if (m.op === 'close') { await s.interrupt(); s.close(); sessions.delete(m.session); return {}; }
  throw Error('Unknown operation');
}

createInterface({input: process.stdin}).on('line', line => {
  let m; try { m = JSON.parse(line); } catch { return; }
  Promise.resolve().then(() => handle(m)).then(result => {
    if (m.id !== undefined) send({id: m.id, result: result ?? {}});
  }).catch(error => {
    log('op', m.op, 'failed:', error.message);
    if (m.id !== undefined) send({id: m.id, error: error.code === 'chatgpt_login_required' ? error.code : 'codex_request_failed'});
  });
});
process.stdin.on('end', () => { for (const s of sessions.values()) s.close(); process.exit(0); });
