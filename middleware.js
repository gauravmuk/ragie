import { NextResponse } from "next/server";

const WINDOW_MS = 60_000;
const MAX_REQUESTS = 10;
const rateLimitMap = new Map();

function cleanupStaleEntries() {
  const now = Date.now();
  for (const [key, entry] of rateLimitMap) {
    if (now > entry.resetAt) rateLimitMap.delete(key);
  }
}

export function middleware(request) {
  if (request.nextUrl.pathname !== "/api/query") {
    return NextResponse.next();
  }

  if (rateLimitMap.size > 10_000) cleanupStaleEntries();

  const ip = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
  const now = Date.now();
  const entry = rateLimitMap.get(ip) ?? { count: 0, resetAt: now + WINDOW_MS };

  if (now > entry.resetAt) {
    entry.count = 0;
    entry.resetAt = now + WINDOW_MS;
  }

  entry.count += 1;
  rateLimitMap.set(ip, entry);

  if (entry.count > MAX_REQUESTS) {
    return new NextResponse(
      JSON.stringify({ error: "Too many requests. Please wait a minute." }),
      { status: 429, headers: { "Content-Type": "application/json" } }
    );
  }

  return NextResponse.next();
}

export const config = {
  matcher: "/api/:path*",
};
