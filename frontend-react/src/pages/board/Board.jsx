import React, { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import BoardHeader from "../../components/layout/BoardHeader";
import BoardWrite from "./BoardWrite";
import { BOARD_TYPES } from "./BoardTypes";
import { api } from "../../api/client";

export default function Board() {
  const [boardId, setBoardId] = useState(2);
  const [boards, setBoards] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const nav = useNavigate();

  const title = useMemo(
    () => BOARD_TYPES.find((b) => b.id === boardId)?.name ?? "게시판",
    [boardId]
  );

  const fetchBoards = async () => {
    try {
      setLoading(true);
      setErrorMsg("");

      const res = await api.get("/api/boards", { params: { boardId } });
      setBoards(res.data);
    } catch (e) {
      console.error(e);
      if (e?.response?.status === 401) setErrorMsg("로그인이 필요합니다.");
      else setErrorMsg("목록 불러오기 실패. / 백엔드, DB 확인");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBoards();
  }, [boardId]);

  // 백이 필터링 안 해줘도 프론트에서 한번 더 필터링 (안전장치)
  const visibleBoards = useMemo(() => {
    return (boards ?? []).filter((b) => {
      const bid = b.boardId ?? b.boardTypeId; // 백 필드명에 맞춰
      return Number(bid) === Number(boardId) || bid == null; 
      // bid가 아예 없으면 일단 그대로 보여주게 처리
    });
  }, [boards, boardId]);

  return (
    <>
      <BoardHeader boardId={boardId} setBoardId={setBoardId} title={title} />

      <div className="max-w-4xl mx-auto p-6">
        <BoardWrite boardId={boardId} onSuccess={fetchBoards} />

        {errorMsg && (
          <div className="mt-6 border rounded-xl p-4 bg-red-50 text-red-700">
            {errorMsg}
          </div>
        )}

        <ul className="border rounded-2xl divide-y mt-6 bg-white">
          {loading ? (
            <li className="p-10 text-center text-gray-500">불러오는 중...</li>
          ) : visibleBoards.length === 0 ? (
            <li className="p-10 text-center text-gray-500">아직 글이 없음</li>
          ) : (
            visibleBoards.map((board) => (
              <li
                key={board.id}
                className="p-4 hover:bg-gray-50 cursor-pointer"
                onClick={() => nav(`/board/${board.id}`)}
              >
                <div className="font-medium">{board.title}</div>

                <div className="text-sm text-gray-500 flex flex-wrap gap-4 mt-1">
                  {/* 게시판명 표시(선택) */}
                  <span>게시판: {title}</span>

                  {/* 작성자 표시 */}
                  <span>
                    작성자: {board.author?.loginId ?? board.authorLoginId ?? "알 수 없음"}
                  </span>

                  <span>작성일: {article.regDate}</span>
                  <span>수정일: {article.updateDate}</span>
                </div>
              </li>
            ))
          )}
        </ul>
      </div>
    </>
  );
}
