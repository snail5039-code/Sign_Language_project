import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../../api/client";
import { useModal } from "../../context/ModalContext";

export default function BoardModify() {
  const { id } = useParams();
  const nav = useNavigate();
  const { showModal } = useModal();

  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.get(`/boards/${id}`).then((res) => {
      setTitle(res.data.title ?? "");
      setContent(res.data.content ?? "");
    }).catch(err => {
      console.error(err);
      showModal({ title: "오류", message: "게시글을 불러오지 못했습니다.", type: "error", onClose: () => nav("/board") });
    });
  }, [id]);

  const onSave = async (e) => {
    e.preventDefault();
    const t = title.trim();
    const c = content.trim();

    if (!t || !c) {
      showModal({ title: "입력 오류", message: "제목과 내용을 모두 입력해주세요.", type: "warning" });
      return;
    }

    try {
      setLoading(true);
      await api.put(`/boards/${id}`, { title: t, content: c });
      showModal({
        title: "수정 완료",
        message: "게시글이 성공적으로 수정되었습니다.",
        type: "success",
        onClose: () => nav(`/board/${id}`)
      });
    } catch (e) {
      console.error(e);
      showModal({
        title: "수정 실패",
        message: e?.response?.data?.message || "글 수정 중 오류가 발생했습니다.",
        type: "error"
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => nav(-1)}
          className="mb-8 flex items-center gap-2 text-slate-400 font-black hover:text-indigo-600 transition-colors group"
        >
          <svg className="w-5 h-5 transition-transform group-hover:-translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
          </svg>
          뒤로 가기
        </button>

        <div className="glass rounded-[3rem] p-12 border-slate-100 shadow-2xl animate-fade-in">
          <div className="mb-10">
            <h1 className="text-3xl font-black text-slate-800 tracking-tight">게시글 수정</h1>
            <p className="text-slate-400 mt-2 font-bold">내용을 수정하고 저장하세요.</p>
          </div>

          <form onSubmit={onSave} className="space-y-6">
            <div>
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">제목</label>
              <input
                className="w-full px-6 py-4 bg-white border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 outline-none transition-all placeholder-slate-300 font-bold"
                placeholder="제목을 입력하세요"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                disabled={loading}
              />
            </div>

            <div>
              <label className="block text-sm font-black text-slate-700 mb-2 ml-1">내용</label>
              <textarea
                className="w-full px-6 py-4 bg-white border border-slate-100 rounded-2xl focus:ring-2 focus:ring-indigo-500 outline-none transition-all placeholder-slate-300 font-bold min-h-[400px] resize-none"
                placeholder="내용을 입력하세요"
                value={content}
                onChange={(e) => setContent(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="flex gap-4 pt-4">
              <button
                type="button"
                onClick={() => nav(-1)}
                className="flex-1 py-5 bg-white border border-slate-200 text-slate-600 rounded-2xl font-black hover:bg-slate-50 transition-all active:scale-95"
              >
                취소
              </button>
              <button
                type="submit"
                disabled={loading}
                className="flex-[2] py-5 bg-indigo-600 text-white rounded-2xl font-black shadow-xl shadow-indigo-100 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all disabled:opacity-60 active:scale-95"
              >
                {loading ? "저장 중..." : "수정사항 저장하기"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
