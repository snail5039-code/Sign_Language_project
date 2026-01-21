import React, { useMemo, useState } from "react";
import { api } from "../../api/client";
import CommentWrite from "./CommentWrite";
import LikeButton from "../common/LikeButton";
import { useTranslation } from "react-i18next";

function cn(...xs) {
  return xs.filter(Boolean).join(" ");
}

function initials(name = "") {
  const s = String(name).trim();
  if (!s) return "?";
  return s.slice(0, 1).toUpperCase();
}

export default function CommentItem({ comment, relTypeCode, relId, onRefresh }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(comment.content);
  const [isReplying, setIsReplying] = useState(false);
  const { t } = useTranslation("board");

  const isReply = Boolean(comment.parentId);

  const meta = useMemo(() => {
    return {
      writer: comment.writerName,
      date: comment.updateDate,
    };
  }, [comment.writerName, comment.updateDate]);

  const handleUpdate = async () => {
    try {
      await api.put(`/comments/${comment.id}`, { content: editContent });
      setIsEditing(false);
      onRefresh();
    } catch (e) {
      console.error(e);
      alert(t("comment.editFail"));
    }
  };

  const handleDelete = async () => {
    if (!window.confirm(t("comment.confirmDelete"))) return;
    try {
      await api.delete(`/comments/${comment.id}`);
      onRefresh();
    } catch (e) {
      console.error(e);
      alert(t("comment.deleteFail"));
    }
  };

  return (
    <div
      className={cn(
        "rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] p-4",
        isReply && "ml-10 relative"
      )}
    >
      {isReply && (
        <div className="absolute -left-5 top-4 bottom-4 w-[2px] bg-[var(--border)] rounded-full" />
      )}

      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="h-9 w-9 shrink-0 rounded-2xl border border-[var(--border)] bg-[var(--surface)] flex items-center justify-center text-xs text-white">
            {initials(meta.writer)}
          </div>

          <div className="min-w-0">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-sm font-semibold text-white truncate">{meta.writer}</span>
              <span className="text-xs text-[var(--muted)] whitespace-nowrap">{meta.date}</span>

              <LikeButton
                targetId={comment.id}
                targetType="comment"
                initialLiked={comment.isLiked}
                initialCount={comment.likeCount}
              />
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {!isReply && (
            <button
              onClick={() => setIsReplying(!isReplying)}
              className="text-xs text-[var(--accent)] hover:text-white transition-colors"
            >
              {t("comment.reply")}
            </button>
          )}

          {comment.canModify && (
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="text-xs text-[var(--muted)] hover:text-white transition-colors"
            >
              {isEditing ? t("comment.cancel") : t("comment.edit")}
            </button>
          )}

          {comment.canDelete && (
            <button
              onClick={handleDelete}
              className="text-xs text-rose-300 hover:text-rose-200 transition-colors"
            >
              {t("comment.delete")}
            </button>
          )}
        </div>
      </div>

      {isEditing ? (
        <div className="mt-3 space-y-2">
          <textarea
            className={cn(
              "w-full rounded-2xl border border-[var(--border)] bg-[var(--surface)] px-4 py-3 text-sm text-white",
              "outline-none resize-none placeholder:text-[var(--muted)]",
              "focus:ring-2 focus:ring-[var(--accent)]/35 focus:border-[var(--accent)]/40"
            )}
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            rows={2}
          />
          <div className="flex justify-end">
            <button
              onClick={handleUpdate}
              className="rounded-xl bg-[var(--accent)] px-4 py-2 text-xs font-semibold text-white hover:bg-[var(--accent-strong)] transition-all"
            >
              {t("comment.save")}
            </button>
          </div>
        </div>
      ) : (
        <p className="mt-3 text-sm text-slate-100/90 whitespace-pre-wrap">{comment.content}</p>
      )}

      {isReplying && (
        <div className="mt-4">
          <CommentWrite
            relTypeCode={relTypeCode}
            relId={relId}
            parentId={comment.id}
            onSuccess={() => {
              setIsReplying(false);
              onRefresh();
            }}
            onCancel={() => setIsReplying(false)}
          />
        </div>
      )}
    </div>
  );
}
