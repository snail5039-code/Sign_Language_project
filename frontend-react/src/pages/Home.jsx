import React from "react";
import HomeSidebar from "../components/home/HomeSidebar";

export default function Home() {
  const languages = [
    { name: "영어", code: "en" },
    { name: "한국어", code: "ko" },
    { name: "일본어", code: "ja" }
  ];

  return (
    <div className="min-h-[calc(100-5rem)] bg-slate-50 flex">
      {/* 메인 콘텐츠 영역 */}
      <main className="flex-1 p-8 flex flex-col gap-8">
        {/* 화면 전환 영역 (Large Area) */}
        <div className="flex-1 glass rounded-[3rem] flex items-center justify-center relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 group-hover:opacity-100 transition-opacity"></div>
          <div className="text-center z-10">
            <div className="w-24 h-24 bg-indigo-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-indigo-200 animate-pulse">
              <svg className="w-12 h-12 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <h2 className="text-4xl font-black text-slate-800 tracking-tight mb-2">화면 전환</h2>
            <p className="text-slate-500 font-bold">실시간 수어 번역 및 영상 통화 시스템</p>
          </div>
        </div>

        {/* 하단 언어 버튼 */}
        <div className="flex justify-center gap-6">
          {languages.map((lang) => (
            <button
              key={lang.code}
              className="px-12 py-5 glass rounded-3xl text-xl font-black text-slate-700 hover:bg-indigo-600 hover:text-white hover:-translate-y-1 transition-all shadow-lg active:scale-95"
            >
              {lang.name}
            </button>
          ))}
        </div>
      </main>

      {/* 우측 사이드바 */}
      <HomeSidebar />
    </div>
  );
}
