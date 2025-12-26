import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate, useParams } from "react-router-dom";

export default function BoardDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const [article, setArticle] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (!id) return;

    (async () => {
      try {
        setErrorMsg("");
        const res = await axios.get(`/api/boards/${id}`);
        setArticle(res.data);
      } catch (e) {
        console.error(e);
        setErrorMsg("상세를 불러오지 못했어. 존재하지 않는 글일 수 있어!");
      }
    })();
  }, [id]);

  if (errorMsg) {
    return (
      <div className="max-w-3xl mx-auto p-6">
        <div className="border rounded-xl p-4 bg-red-50 text-red-700">
          {errorMsg}
        </div>
        <button className="mt-4 px-4 py-2 rounded-xl border" onClick={() => nav("/boards")}>
          목록으로
        </button>
      </div>
    );
  }

  if (!article) return <div className="p-10 text-center">로딩중...</div>;

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="bg-white border rounded-2xl p-6">
        <h1 className="text-2xl font-extrabold">{article.title}</h1>
        <div className="text-sm text-gray-500 mt-2 flex gap-4">
          <span>작성일: {article.regDate}</span>
          <span>수정일: {article.updateDate}</span>
        </div>
        <div className="mt-6 whitespace-pre-wrap">{article.content}</div>
        <button className="mt-6 px-4 py-2 rounded-xl border" onClick={() => nav("/board")}>
          목록
        </button>
      </div>
    </div>
  );
}
