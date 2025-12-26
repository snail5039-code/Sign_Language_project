import React, { useState } from "react";
import axios from "axios";

export default function BoardWrite({ boardId, onSuccess }) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!title.trim() || !content.trim()) return;

    try {
      setLoading(true);
      await axios.post("/api/boards", {
        boardId: Number(boardId),
        title: title.trim(),
        content: content.trim(),
      });
      setTitle("");
      setContent("");
      onSuccess?.();
    } catch (e) {
      console.log("✅ status:", e.response?.status);
      console.log("✅ data:", e.response?.data);
      console.log("✅ payload:", {
        boardId,
        title: title.trim(),
        content: content.trim(),
      });
      console.log("✅ full error:", e);
      alert("글 등록 실패. 콘솔의 status/data 확인!");
      }

  };

  return (
    <div className="border rounded-2xl p-5 mt-6 bg-white">
      <input
        className="w-full border p-3 mb-3 rounded-xl"
        placeholder="제목"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        disabled={loading}
      />
      <textarea
        className="w-full border p-3 mb-3 rounded-xl min-h-[120px]"
        placeholder="내용"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        disabled={loading}
      />
      <button
        onClick={handleSubmit}
        disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded-xl disabled:opacity-60"
      >
        {loading ? "등록 중..." : "글 등록"}
      </button>
    </div>
  );
}
