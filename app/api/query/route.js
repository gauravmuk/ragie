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
    const result = await querySystem(question.trim(), { userId, sessionId });

    const sources = result.reranked.slice(0, 5).map((item) => ({
      title: item.doc.metadata?.title ?? "Untitled",
      source: item.doc.metadata?.source ?? null,
      facet: item.docFacet,
    }));

    return Response.json({
      answer: result.answer,
      facet: result.requestedFacet,
      noContext: result.noContext,
      sources,
    });
  } catch (err) {
    console.error("Query API error:", err.message, err.stack);
    return Response.json(
      { error: "An error occurred. Please try again." },
      { status: 500 }
    );
  }
}
