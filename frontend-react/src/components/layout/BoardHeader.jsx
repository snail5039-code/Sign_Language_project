import React from "react";
import { BOARD_TYPES } from "../../pages/board/BoardTypes";
import { Search } from "lucide-react";


export default function BoardHeader({ boardId, setBoardId, keyword, setKeyword, onSearch }) {
  return (
    <div className="board">
      <div className="max-w-5xl mx-auto px-6 py-14">
        <h1 className="text-4xl font-extrabold text-center">게시판</h1>

        {/* 검색바 */}
        {/* <div className="mt-10">
          <div className="relative max-w-3xl mx-auto">
            <input
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSearch?.()}
              placeholder="단어, 문장을 검색하세요"
              className="w-full h-14 rounded-full bg-white border border-gray-200 px-6 pr-14 outline-none focus:ring-2 focus:ring-blue-200"
            />
            <button
              type="button"
              onClick={() => onSearch?.()}
              className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full hover:bg-gray-100 flex items-center justify-center"
            >
              <Search className="w-5 h-5 text-blue-600" />
            </button>
          </div>
        </div> */}

        {/* 칩(탭) */}
        <div className="mt-10 flex justify-center gap-6 flex-wrap">
          {BOARD_TYPES.map((b) => {
            const active = b.id === boardId;

            return (
              <button
                key={b.id}
                onClick={() => setBoardId(b.id)}
                className="flex flex-col items-center gap-2"
              >
                <div
                  className={[
                    "w-20 h-20 rounded-2xl border flex items-center justify-center transition",
                    active ? "bg-blue-600 border-blue-600" : "bg-white border-gray-200 hover:bg-gray-50",
                  ].join(" ")}
                >
                  {/* 아이콘 없어도 됨. 원하면 여기 넣기 */}
                  <span className={active ? "text-white font-bold" : "text-blue-600 font-bold"}>
                    {b.name.slice(0, 2)} {/* 몇 글자 넣을지 */}
                  </span>
                </div>
                <div className={active ? "text-sm font-semibold" : "text-sm text-gray-700"}>
                  {b.name}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}