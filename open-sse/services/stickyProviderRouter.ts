import * as crypto from "crypto";

// Session → Provider binding cache
const sessionProviders = new Map<string, {
  providerId: string;
  accountEmail: string;
  boundAt: number;
  expiresAt: number;
}>();

const MAX_SESSIONS = 1000;
const TTL_MS = 1000 * 60 * 60 * 24;  // 24hr

// Auto-prune expired sessions every 5 minutes
setInterval(() => {
  const now = Date.now();
  let pruned = 0;
  for (const [sessionId, binding] of sessionProviders.entries()) {
    if (binding.expiresAt <= now) {
      sessionProviders.delete(sessionId);
      pruned++;
    }
  }
  if (pruned > 0) {
    console.log(`[Sticky Session] Pruned ${pruned} expired sessions`);
  }
}, 1000 * 60 * 5);  // Every 5 minutes

export function bindSessionToProvider(sessionId: string, providerId: string, accountEmail: string) {
  // Prune if cache too large
  if (sessionProviders.size >= MAX_SESSIONS) {
    const oldest = [...sessionProviders.entries()]
      .sort((a, b) => a[1].boundAt - b[1].boundAt)[0];
    if (oldest) {
      sessionProviders.delete(oldest[0]);
      console.warn(`[Sticky Session] Cache full, evicted oldest session`);
    }
  }

  // Establish "handshake" - lock this conversation to this account
  sessionProviders.set(sessionId, {
    providerId,
    accountEmail,
    boundAt: Date.now(),
    expiresAt: Date.now() + TTL_MS,
  });
  console.log(`[Sticky Session] Bound session ${sessionId} to account ${accountEmail}`);
}

export function getSessionProvider(sessionId: string): string | null {
  // Return bound provider (the "key" for this conversation's "lock")
  const binding = sessionProviders.get(sessionId);
  if (!binding) return null;

  if (binding.expiresAt <= Date.now()) {
    sessionProviders.delete(sessionId);
    return null;
  }

  return binding.providerId;
}

export function releaseSession(sessionId: string) {
  // Unlock conversation - allow rotation again
  sessionProviders.delete(sessionId);
  console.log(`[Sticky Session] Released session ${sessionId}`);
}

export function hasActiveSignatures(body: any): boolean {
  // Check if request contains tool_calls with cached signatures
  const messages = body.messages || [];
  for (const msg of messages) {
    if (msg.role === "assistant" && msg.tool_calls) {
      return true;  // Has tool calls, needs sticky routing
    }
  }
  return false;
}

export function extractSessionId(body: any, headers: any): string {
  // Hash first N messages + client fingerprint to prevent collisions
  const messages = body.messages || [];
  const sessionMessages = messages.slice(0, 5);  // first 5 messages

  // Include client fingerprint to prevent "Hello" collision between users
  const authHeader = headers.authorization || headers.Authorization || '';
  const clientFingerprint = authHeader.slice(-16);  // last 16 chars of API key

  const hashInput = JSON.stringify({
    messages: sessionMessages,
    fingerprint: clientFingerprint,
  });

  return crypto.createHash('sha256')
    .update(hashInput)
    .digest('hex')
    .slice(0, 16);
}
