import React, { useState } from "react";
import axios from "axios";
import { BOARD_TYPES } from "./BoardTypes";

export default function BoardWrite({ onSuccess }) {
    const [boardId, setBoardId] = useState(2);
    const [title, setTitle] = useState("");
    const [content, setContent] = useState("");

    const handleSubmit = async () => {
        if (!title || !content) return;

        await axios.post("api/boards", {
            boardId,
            title,
            content
        });

        setTitle("");
        setContent("");
        onSuccess();
    };

    return (
        <div className="border rounded p-4 mb-6">
            <select 
                className="border p-2 mb-2 rounded w-full"
                value={boardId}
                onChange={(e) => setBoardId(Number(e.target.value))}>
                    {BOARD_TYPES.map((board) => (
                        <option key={board.id} value={board.id}>
                            {board.name}
                        </option>
                    ))}
                </select>
                <input 
                    className="w-full border p-2 mb-2 rounded"
                    placeholder="제목"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                 />
                 <input 
                    className="w-full border p-2 mb-2 rounded"
                    placeholder="내용"
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                 />
                 <button 
                    onClick={handleSubmit}
                    className="bg-balck text-white px-4 py-2 reunded">
                    글 등록
                 </button>
        </div>
    );
}