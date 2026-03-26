import { querySystem } from "../../../query.js";

export const maxDuration = 60;

const MAX_QUESTION_LENGTH = 500;

export async function POST(request) {
  const body = await request.json();
  const question = body?.question;

  if (!question || typeof question !== "string" || question.trim().length === 0) {
    return Response.json({ error: "Question is required" }, { status: 400 });
  }

  if (question.length > MAX_QUESTION_LENGTH) {
    return Response.json(
      { error: `Question must be ${MAX_QUESTION_LENGTH} characters or fewer` },
      { status: 400 }
    );
  }

  const userId = typeof body?.userId === "string" ? body.userId.slice(0, 128) : undefined;
  const sessionId = typeof body?.sessionId === "string" ? body.sessionId.slice(0, 128) : undefined;

  try {
    const result = await querySystem(question.trim(), {
      userId,
      sessionId,
      streaming: true,
    });

    const seenSources = new Set();
    const sources = [];
    for (const item of result.reranked) {
      const source = item.doc.metadata?.source || null;
      if (source && seenSources.has(source)) continue;
      if (source) seenSources.add(source);

      sources.push({
        title: item.doc.metadata?.title ?? "Untitled",
        source: source,
        facet: item.docFacet,
      });
      if (sources.length >= 5) break;
    }

    if (result.noContext) {
      return Response.json({
        answer: "",
        facet: result.requestedFacet,
        noContext: true,
        sources: [],
      });
    }

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        // Send initial metadata as the first chunk
        const metadata = {
          facet: result.requestedFacet,
          sources,
          noContext: false,
        };
        controller.enqueue(encoder.encode(`__METADATA__:${JSON.stringify(metadata)}\n`));

        for await (const chunk of result.stream) {
          const content = typeof chunk.content === "string" ? chunk.content : JSON.stringify(chunk.content);
          if (content) {
            controller.enqueue(encoder.encode(content));
          }
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Transfer-Encoding": "chunked",
      },
    });
  } catch (err) {
    console.error("Query API error:", err.message, err.stack);
    return Response.json(
      { error: "An error occurred. Please try again." },
      { status: 500 }
    );
  }
}
