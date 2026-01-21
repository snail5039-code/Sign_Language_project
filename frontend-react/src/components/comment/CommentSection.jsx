import React, { useEffect, useState, useCallback } from "react";
import { api } from "../../api/client";
import CommentItem from "./CommentItem";
import CommentWrite from "./CommentWrite";
import { useTranslation } from "react-i18next";

export default function CommentSection({ relTypeCode = "article", relId }) {
  const [comments, setComments] = useState([]);
  const [loading, setLoading] = useState(false);
  const { t } = useTranslation("board");

  const fetchComments = useCallback(async () => {
    try {
      setLoading(true);
      const res = await api.get(`/comments/${relTypeCode}/${relId}`);
      setComments(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [relTypeCode, relId]);

  useEffect(() => {
    if (relId) fetchComments();
  }, [relId, fetchComments]);

  const rootComments = comments.filter((c) => !c.parentId);
  const getReplies = (parentId) => comments.filter((c) => c.parentId === parentId);

  return (
    <div className="mt-10">
      <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[0_18px_40px_rgba(6,12,26,0.35)]">
        <div className="flex items-end justify-between gap-4">
          <h3 className="text-base font-semibold text-white">
            {t("comment.count", { count: comments.length })}
          </h3>
          <span className="text-xs text-[var(--muted)]">{t("comment.guide", { defaultValue: "" })}</span>
        </div>

        {/* 작성 */}
        <CommentWrite relTypeCode={relTypeCode} relId={relId} onSuccess={fetchComments} />

        {/* 목록 */}
        <div className="mt-6 space-y-3">
          {loading && comments.length === 0 ? (
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-6 text-center text-sm text-[var(--muted)]">
              {t("comment.loading")}
            </div>
          ) : rootComments.length === 0 ? (
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-6 text-center text-sm text-[var(--muted)]">
              {t("comment.empty")}
            </div>
          ) : (
            rootComments.map((comment) => (
              <div key={comment.id} className="space-y-3">
                <CommentItem
                  comment={comment}
                  relTypeCode={relTypeCode}
                  relId={relId}
                  onRefresh={fetchComments}
                />

                {getReplies(comment.id).map((reply) => (
                  <CommentItem
                    key={reply.id}
                    comment={reply}
                    relTypeCode={relTypeCode}
                    relId={relId}
                    onRefresh={fetchComments}
                  />
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
