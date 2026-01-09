import React, { useEffect, useMemo, useState } from "react";
import { useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../../api/client";
import { useModal } from "../../context/ModalContext";
import LikeButton from "../../components/common/LikeButton";
import CommentSection from "../../components/comment/CommentSection";

const BOARD_TYPES = [
  { id: 1, name: "공지사항" },
  { id: 2, name: "자유게시판" },
  { id: 3, name: "QnA" },
  { id: 4, name: "오류사항접수" }
];

export default function BoardDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const { showModal } = useModal();
  const [article, setArticle] = useState(null);
  const lastHitIdRef = useRef(null);

  useEffect(() => {
    if (!id) return;
    (async () => {
      try {
        const res = await api.get(`/boards/${id}`);
        setArticle(res.data);

        if (lastHitIdRef.current !== id) {
          lastHitIdRef.current = id;
          try {
            const hitRes = await api.patch(`/boards/${id}/hit`);
            if (hitRes?.data?.hit != null) {
              setArticle((prev) => (prev ? { ...prev, hit: hitRes.data.hit } : prev));
            } else {
              setArticle((prev) => (prev ? { ...prev, hit: (prev.hit || 0) + 1 } : prev));
            }
          } catch (hitErr) {
            console.warn("View count update failed", hitErr);
          }
        }
      } catch (e) {
        console.error(e);
        const status = e?.response?.status;
        const message = status === 404
          ? "삭제되었거나 존재하지 않는 게시글입니다."
          : "게시글을 불러오지 못했습니다.";
        showModal({
          title: "오류",
          message,
          type: "error",
          onClose: () => nav("/board", { replace: true })
        });
      }
    })();
  }, [id, nav, showModal]);

  const boardName = useMemo(() => {
    const typeId = article?.boardId ?? article?.boardTypeId;
    return BOARD_TYPES.find((b) => b.id === Number(typeId))?.name ?? "게시판";
  }, [article]);

  const handleDelete = () => {
    showModal({
      title: "게시글 삭제",
      message: "정말로 게시글을 삭제하시겠습니까?",
      type: "warning",
      children: (
        <div className="flex gap-3">
          <button
            onClick={async () => {
              try {
                await api.delete(`/boards/${id}`);
                nav("/board", { replace: true });
                showModal({
                  title: "삭제 완료",
                  message: "게시글이 성공적으로 삭제되었습니다.",
                  type: "success"
                });
              } catch (e) {
                showModal({
                  title: "삭제 실패",
                  message: e?.response?.data?.message || "삭제 중 오류가 발생했습니다.",
                  type: "error"
                });
              }
            }}
            className="flex-1 py-4 bg-rose-600 text-white rounded-2xl font-black hover:bg-rose-700 transition-all shadow-lg shadow-rose-100"
          >
            삭제하기
          </button>
        </div>
      )
    });
  };

  if (!article) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
        <div className="w-12 h-12 border-4 border-[var(--accent)] border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg)] p-8 text-[var(--text)]">
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => nav("/board")}
          className="mb-8 flex items-center gap-2 text-slate-300 font-black hover:text-[var(--accent)] transition-colors group"
        >
          <svg className="w-5 h-5 transition-transform group-hover:-translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
          </svg>
          목록으로 돌아가기
        </button>

        <div className="rounded-[3rem] overflow-hidden border border-[var(--border)] bg-[rgba(18,27,47,0.94)] shadow-2xl animate-fade-in">
          <div className="p-12">
            <div className="flex items-center gap-3 mb-6">
              <span className="px-4 py-1.5 bg-[var(--accent)]/20 text-[var(--accent)] text-xs font-black rounded-full border border-[rgba(59,130,246,0.35)] uppercase tracking-widest">
                {boardName}
              </span>
              <span className="text-sm font-bold text-slate-300">#{article.id}</span>
            </div>

            <h1 className="text-4xl font-black text-slate-100 tracking-tight mb-8 leading-tight">
              {article.title}
            </h1>

            <div className="flex items-center justify-between pb-8 border-b border-[var(--border)] mb-8">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-[var(--surface-soft)] rounded-2xl flex items-center justify-center text-slate-300">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <div className="text-base font-black text-slate-100">{article.writerName || "익명"}</div>
                  <div className="text-xs font-bold text-slate-400">{article.regDate}</div>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <div className="text-xs font-black text-slate-300 uppercase tracking-widest mb-1">Views</div>
                  <div className="text-lg font-black text-slate-100">{article.hit || 0}</div>
                </div>
              </div>
            </div>

            <div className="prose prose-invert max-w-none min-h-[300px] text-slate-100 font-bold leading-relaxed whitespace-pre-wrap">
              {article.content}
            </div>

            <div className="mt-12 pt-12 border-t border-[var(--border)] flex items-center justify-between">
              <LikeButton
                targetId={id}
                targetType="article"
                initialLiked={article.isLiked}
                initialCount={article.likeCount}
              />

              <div className="flex gap-3">
                {article.canModify && (
                  <button
                    onClick={() => nav(`/board/${id}/modify`)}
                    className="px-6 py-3 bg-[var(--surface-soft)] border border-[var(--border)] text-slate-100 rounded-2xl font-black hover:bg-[var(--surface)] transition-all shadow-sm active:scale-95"
                  >
                    수정하기
                  </button>
                )}
                {article.canDelete && (
                  <button
                    onClick={handleDelete}
                    className="px-6 py-3 bg-rose-500/10 text-rose-300 border border-rose-500/30 rounded-2xl font-black hover:bg-rose-500/20 transition-all shadow-sm active:scale-95"
                  >
                    삭제하기
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="bg-[var(--surface-soft)] p-12 border-t border-[var(--border)]">
            <CommentSection relTypeCode="article" relId={id} />
          </div>
        </div>
      </div>
    </div>
  );
}











