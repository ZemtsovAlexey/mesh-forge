import type { Artifact, ChatMessage, ReplyTarget } from "./types";

const VIEW_LABELS: Record<string, string> = {
  front: "спереди",
  left: "слева",
  right: "справа",
  back: "сзади",
  preview: "превью",
};

export function artifactCaption(art: Artifact): string {
  const key = (art.label || art.view || "").toLowerCase();
  return VIEW_LABELS[key] || art.label || "";
}

export function collectMessageArtifacts(message: ChatMessage): Artifact[] {
  const items: Artifact[] = [];
  const seen = new Set<string>();
  for (const art of [...(message.attachments || []), ...(message.artifacts || [])]) {
    const key = art.id || art.name;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    items.push(art);
  }
  for (const tool of message.tools || []) {
    for (const art of tool.artifacts || []) {
      const key = art.id || art.name;
      if (!key || seen.has(key)) continue;
      seen.add(key);
      items.push(art);
    }
  }
  return items;
}

export function replyPreview(message: ChatMessage, artifactIds?: string[]): string {
  const arts = collectMessageArtifacts(message);
  const wanted = artifactIds?.length ? arts.filter((a) => artifactIds.includes(a.id)) : arts;
  if (wanted.length === 1) {
    const art = wanted[0];
    if (art.kind === "mesh") return "mesh";
    const cap = artifactCaption(art);
    return cap ? `вид ${cap}` : "картинку";
  }
  if (wanted.length > 1) {
    const views = wanted.map(artifactCaption).filter(Boolean);
    if (views.length === wanted.length) return views.join(", ");
    return `${wanted.length} вида`;
  }
  const text = (message.content || "").trim().replace(/\s+/g, " ");
  if (text) return text.length > 72 ? `${text.slice(0, 71)}…` : text;
  return "сообщение";
}

export function makeReplyTarget(message: ChatMessage, artifact?: Artifact): ReplyTarget {
  const artifactIds = artifact ? [artifact.id] : collectMessageArtifacts(message).map((a) => a.id);
  return {
    messageId: message.id,
    preview: replyPreview(message, artifact ? [artifact.id] : artifactIds),
    artifactIds,
  };
}
