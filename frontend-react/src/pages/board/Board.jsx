import React, { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../../api/client";
import { useModal } from "../../context/ModalContext";
import { BOARD_TYPES } from "./BoardTypes";

const SORT_OPTIONS = [
  { value: "latest", label: "최신순" },
  { value: "views", label: "조회수" },
  { value: "comments", label: "댓글순" }
];

const getStoredBoardId = () => {
  if (typeof window === "undefined") return 2;
  const stored = Number(window.sessionStorage.getItem("boardId"));
  return !Number.isNaN(stored) && stored > 0 ? stored : 2;
};

const resolveBoardId = (typeParam) => {
  if (!typeParam) return getStoredBoardId();
  const numeric = Number(typeParam);
  if (!Number.isNaN(numeric) && numeric > 0) return numeric;
  return BOARD_TYPES.find((b) => b.key === typeParam)?.id ?? getStoredBoardId();
};

const resolveBoardKey = (id) => BOARD_TYPES.find((b) => b.id === id)?.key ?? String(id);

export default function Board() {
  const { showModal } = useModal();
  const [searchParams, setSearchParams] = useSearchParams();

  const [cPage, setCPage] = useState(1);
  const [boardId, setBoardId] = useState(() => resolveBoardId(searchParams.get("type")));
  const [boards, setBoards] = useState([]);
  const [searchInput, setSearchInput] = useState("");
  const [searchKeyword, setSearchKeyword] = useState("");
  const [searchType, setSearchType] = useState("title");
  const [pageSize, setPageSize] = useState(10);
  const [sortType, setSortType] = useState("latest");

  const [pageInfo, setPageInfo] = useState({
    totalPagesCnt: 1,
    begin: 1,
    end: 1,
  });
  const [loading, setLoading] = useState(false);

  const nav = useNavigate();

  useEffect(() => {
    const nextId = resolveBoardId(searchParams.get("type"));
    if (nextId !== boardId) {
      setBoardId(nextId);
      setCPage(1);
    }
  }, [searchParams, boardId]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.sessionStorage.setItem("boardId", String(boardId));
    }
  }, [boardId]);

  useEffect(() => {
    setSearchInput(searchKeyword);
  }, [searchKeyword]);

  const fetchBoards = async () => {
    try {
      setLoading(true);
      const res = await api.get("/boards", {
        params: {
          boardId,
          cPage,
          searchType,
          searchKeyword,
          pageSize,
          sortType
        }
      });

      const totalPagesCnt = res.data.totalPagesCnt ?? 1;
      const begin = res.data.begin ?? 1;
      const end = res.data.end ?? totalPagesCnt;

      setBoards(res.data.articles || []);
      setPageInfo({ totalPagesCnt, begin, end });

      if (cPage > totalPagesCnt) {
        setCPage(totalPagesCnt);
      }
    } catch (e) {
      console.error(e);
      showModal({ title: "오류", message: "게시글을 불러오는 중 오류가 발생했습니다.", type: "error" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBoards();
  }, [boardId, cPage, pageSize, searchType, searchKeyword, sortType]);

  useEffect(() => {
    setCPage(1);
  }, [boardId, pageSize, searchType, searchKeyword, sortType]);

  const handleBoardChange = (id) => {
    setBoardId(id);
    setCPage(1);
    const next = new URLSearchParams(searchParams);
    next.set("type", resolveBoardKey(id));
    setSearchParams(next, { replace: true });
  };

  const handleSearch = () => {
    setSearchKeyword(searchInput.trim());
    setCPage(1);
  };

  const pageNumbers = useMemo(() => {
    const pages = [];
    for (let i = pageInfo.begin; i <= pageInfo.end; i++) {
      pages.push(i);
    }
    return pages.length > 0 ? pages : [1];
  }, [pageInfo]);

  const formatDate = (dateStr) => {
    if (!dateStr) return "-";
    const date = new Date(dateStr);
    return date.toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit" });
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-[1200px] mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <div className="relative group">
              <button className="px-8 py-4 bg-white border border-slate-200 rounded-2xl font-black text-slate-700 flex items-center gap-3 shadow-sm hover:border-indigo-300 transition-all">
                {BOARD_TYPES.find((b) => b.id === boardId)?.name}
                <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              <div className="absolute top-full left-0 mt-2 w-48 bg-white rounded-3xl shadow-2xl border border-slate-100 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all transform origin-top scale-95 group-hover:scale-100 p-2 z-50">
                {BOARD_TYPES.map((type) => (
                  <button
                    key={type.id}
                    onClick={() => handleBoardChange(type.id)}
                    className="w-full text-left px-4 py-3 text-sm font-bold text-slate-600 hover:bg-indigo-50 hover:text-indigo-600 rounded-2xl transition-all"
                  >
                    {type.name}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-sm font-black text-slate-400">정렬</span>
            <select
              value={sortType}
              onChange={(e) => {
                setSortType(e.target.value);
                setCPage(1);
              }}
              className="px-4 py-3 bg-white border border-slate-200 rounded-2xl font-bold text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500 transition-all shadow-sm"
            >
              {SORT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>

            <span className="text-sm font-black text-slate-400">목록</span>
            <select
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setCPage(1);
              }}
              className="px-4 py-3 bg-white border border-slate-200 rounded-2xl font-bold text-slate-600 outline-none focus:ring-2 focus:ring-indigo-500 transition-all shadow-sm"
            >
              <option value={10}>10개씩 보기</option>
              <option value={20}>20개씩 보기</option>
              <option value={30}>30개씩 보기</option>
            </select>
          </div>
        </div>

        <div className="glass rounded-[2.5rem] overflow-hidden border-slate-100 mb-8">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50/50 border-b border-slate-100">
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest w-20">번호</th>
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest">제목</th>
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest w-32">작성자</th>
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest w-32">날짜</th>
                <th className="px-8 py-5 text-xs font-black text-slate-400 uppercase tracking-widest w-24 text-center">조회수</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {loading ? (
                <tr><td colSpan={5} className="px-8 py-20 text-center font-bold text-slate-400">불러오는 중...</td></tr>
              ) : boards.length === 0 ? (
                <tr><td colSpan={5} className="px-8 py-20 text-center font-bold text-slate-400">등록된 게시글이 없습니다.</td></tr>
              ) : (
                boards.map((b) => (
                  <tr
                    key={b.id}
                    onClick={() => nav(`/board/${b.id}`)}
                    className="group hover:bg-indigo-50/30 cursor-pointer transition-colors"
                  >
                    <td className="px-8 py-6 text-sm font-bold text-slate-400">{b.id}</td>
                    <td className="px-8 py-6">
                      <div className="flex items-center gap-3">
                        <span className="text-base font-black text-slate-700 group-hover:text-indigo-600 transition-colors">{b.title}</span>
                        {b.commentCount > 0 && (
                          <span className="px-2 py-0.5 bg-indigo-100 text-indigo-600 text-[10px] font-black rounded-full">
                            {b.commentCount}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-8 py-6 text-sm font-bold text-slate-600">{b.writerName || "익명"}</td>
                    <td className="px-8 py-6 text-sm font-bold text-slate-400">{formatDate(b.regDate)}</td>
                    <td className="px-8 py-6 text-sm font-bold text-slate-400 text-center">{b.hit || 0}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="flex flex-col gap-8">
          <div className="flex justify-center items-center gap-2">
            <button
              onClick={() => setCPage(1)}
              disabled={cPage === 1}
              className="w-10 h-10 flex items-center justify-center rounded-xl border border-slate-200 text-slate-400 hover:bg-white hover:text-indigo-600 disabled:opacity-30 transition-all"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            </button>
            <button
              onClick={() => setCPage(Math.max(1, cPage - 1))}
              disabled={cPage === 1}
              className="w-10 h-10 flex items-center justify-center rounded-xl border border-slate-200 text-slate-400 hover:bg-white hover:text-indigo-600 disabled:opacity-30 transition-all"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
              </svg>
            </button>

            {pageNumbers.map((p) => (
              <button
                key={p}
                onClick={() => setCPage(p)}
                className={`w-10 h-10 flex items-center justify-center rounded-xl font-black text-sm transition-all ${cPage === p
                    ? "bg-indigo-600 text-white shadow-lg shadow-indigo-100"
                    : "text-slate-400 hover:bg-white hover:text-indigo-600"
                  }`}
              >
                {p}
              </button>
            ))}

            <button
              onClick={() => setCPage(Math.min(pageInfo.totalPagesCnt, cPage + 1))}
              disabled={cPage >= pageInfo.totalPagesCnt}
              className="w-10 h-10 flex items-center justify-center rounded-xl border border-slate-200 text-slate-400 hover:bg-white hover:text-indigo-600 disabled:opacity-30 transition-all"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
              </svg>
            </button>
            <button
              onClick={() => setCPage(pageInfo.totalPagesCnt)}
              disabled={cPage >= pageInfo.totalPagesCnt}
              className="w-10 h-10 flex items-center justify-center rounded-xl border border-slate-200 text-slate-400 hover:bg-white hover:text-indigo-600 disabled:opacity-30 transition-all"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
              </svg>
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex-1 flex justify-center">
              <div className="flex items-center gap-2 bg-white p-2 rounded-[2rem] shadow-sm border border-slate-100 w-full max-w-2xl">
                <select
                  value={searchType}
                  onChange={(e) => setSearchType(e.target.value)}
                  className="pl-4 pr-2 py-2 bg-transparent border-none outline-none font-black text-slate-500 text-sm"
                >
                  <option value="title">제목</option>
                  <option value="content">내용</option>
                  <option value="title,content">제목+내용</option>
                </select>
                <div className="h-4 w-[1px] bg-slate-200"></div>
                <input
                  type="text"
                  placeholder="검색어를 입력하세요"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  className="flex-1 px-4 py-2 bg-transparent border-none outline-none font-bold text-slate-700 placeholder-slate-300"
                />
                <button
                  onClick={handleSearch}
                  className="px-6 py-3 bg-slate-900 text-white rounded-[1.5rem] font-black text-sm hover:bg-slate-800 transition-all active:scale-95"
                >
                  검색
                </button>
              </div>
            </div>

            <button
              onClick={() => nav(`/board/write?boardId=${boardId}`)}
              className="px-8 py-4 bg-indigo-600 text-white rounded-2xl font-black shadow-xl shadow-indigo-100 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 4v16m8-8H4" />
              </svg>
              글쓰기
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

