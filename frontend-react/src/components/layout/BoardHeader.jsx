import React from "react";
import { BOARD_TYPES } from "../../pages/board/BoardTypes";

export default function BoardHeader({ boardId, setBoardId }) {
    return (
        <div className="board-6 mb-6">
            <div className="flex gap-4 p-4">
                {BOARD_TYPES.map((board) => (
                 <button
                   key={board.id}
                    onClick={() => setBoardId(board.id)}
                    className={
                      boardId === board.id
                        ? "font-semibold border-b-2 border-black"
                         : "text-gray-500"
                    }
                 >
                    {board.name}
                 </button>
                ))}
            </div>
        </div>
    );
}