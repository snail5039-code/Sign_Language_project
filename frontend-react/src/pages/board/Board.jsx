import React, { useState, useEffect } from "react";
import axios from "axios";
import BoardHeader from "../../components/layout/BoardHeader";
import BoardWrite from "./BoardWrite";
import { BOARD_TYPES } from "./BoardTypes";

export default function Board() {
    const [boardId, setBoardId] = useState(2);
    const [boards, setBoards] = useState([]);

    const fetchBoards = async () => {
        const res = await axios.get(`/api/boards?boardId=${boardId}`);
        setBoards(res.data);
    };

    useEffect(() => {
        fetchBoards();
    }, [boardId]);

    return (
        <div className="max-w-4xl mx-auto p-6">
            <BoardHeader boardId={boardId} setBoardId={setBoardId} />
            <BoardWrite onSuccess={fetchBoards} />

            <h2 className="font-semibold mb-3">
                {BOARD_TYPES.find((b) => b.id === boardId)?.name}
            </h2>

            <ul className="border rounded divide-y">
                {boards.map((board) => (
                    <li key={board.id} className="p-4">
                        <div className="font-medium">{board.title}</div>
                        <div className="text-sm text-gray-500 flex gap-4">
                            <span>작성일: {board.regDate}</span>
                            <span>수정일: {board.updateDate}</span>
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
}