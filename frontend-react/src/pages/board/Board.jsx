import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import BoardHeader from "../../components/layout/BoardHeader";
import BoardWrite from "./BoardWrite";
import { BOARD_TYPES } from "./BoardTypes";

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

      const res = await axios.get("/api/boards", { params: { boardId } });
      setBoards(res.data);
    } catch (e) {
      console.error(e);
      setErrorMsg("목록 불러오기 실패. / 백엔드, DB 확인");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBoards();
  }, [boardId]);

  return (
    <>
      <BoardHeader
        boardId={boardId}
        setBoardId={setBoardId}
        title={title}
      />

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
          ) : boards.length === 0 ? (
            <li className="p-10 text-center text-gray-500">아직 글이 없음</li>
          ) : (
            boards.map((board) => (
              <li
                key={board.id}
                className="p-4 hover:bg-gray-50 cursor-pointer"
                onClick={() => nav(`/board/${board.id}`)}
              >
                <div className="font-medium">{board.title}</div>
                <div className="text-sm text-gray-500 flex gap-4 mt-1">
                  <span>작성일: {board.regDate}</span>
                  <span>수정일: {board.updateDate}</span>
                </div>
              </li>
            ))
          )}
        </ul>
      </div>
    </>
  );
}
