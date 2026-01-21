import React, { useState } from "react";
import { api } from "../../api/client";
import { useTranslation } from "react-i18next";

function cn(...xs) {
  return xs.filter(Boolean).join(" ");
}

export default function CommentWrite({ relTypeCode, relId, parentId = null, onSuccess, onCancel = null }) {
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const { t } = useTranslation("board");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!content.trim()) return;

    try {
      setLoading(true);
      await api.post(`/comments/${relTypeCode}/${relId}`, { content, parentId });
      setContent("");
      if (onSuccess) onSuccess();
    } catch (e) {
      console.error(e);
      alert(e?.response?.data?.message || t("comment.writeFail", { defaultValue: "Failed to post comment" }));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={cn("mt-4", parentId && "mt-3")}>
      <div className="space-y-2">
        <textarea
          className={cn(
            "w-full rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3 text-sm text-white",
            "placeholder:text-[var(--muted)] outline-none resize-none transition-all",
            "focus:ring-2 focus:ring-[var(--accent)]/35 focus:border-[var(--accent)]/40"
          )}
          rows={parentId ? 2 : 3}
          placeholder={parentId ? t("comment.placeholderReply") : t("comment.placeholderComment")}
          value={content}
          onChange={(e) => setContent(e.target.value)}
        />

        <div className="flex justify-end gap-2">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="rounded-xl border border-[var(--border)] bg-transparent px-4 py-2 text-xs text-[var(--muted)] hover:text-white hover:border-[var(--accent)]/40 transition-all"
            >
              {t("comment.cancel")}
            </button>
          )}

          <button
            type="submit"
            disabled={loading || !content.trim()}
            className="rounded-xl bg-[var(--accent)] px-4 py-2 text-xs font-semibold text-white hover:bg-[var(--accent-strong)] transition-all disabled:opacity-50"
          >
            {loading ? t("comment.submitting") : parentId ? t("comment.submitReply") : t("comment.submitComment")}
          </button>
        </div>
      </div>
    </form>
  );
}
