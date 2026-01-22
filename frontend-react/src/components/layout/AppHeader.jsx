import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../auth/AuthProvider";
import useLangMenu from "../../shared/useLangMenu";

export default function AppHeader() {
  const { user, isAuthed, logout } = useAuth();
  const nav = useNavigate();

  const { t } = useTranslation(["header", "common"]);
  const { LANGUAGES, isLangOpen, setIsLangOpen, currentLang, selectLang } =
    useLangMenu();

  const onLogout = async () => {
    await logout();
    nav("/");
  };

  return (
    <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--surface-soft)]/80 backdrop-blur-xl">
      {/* ✅ 헤더 높이 살짝 키움 */}
      <div className="flex h-[72px] items-center">
        {/* ✅ 왼쪽 영역 = 사이드바 폭(w-72)과 맞춰서 브랜드 고정 */}
        <div className="hidden lg:flex w-72 h-full items-center border-r border-[var(--border)] px-6">
          <Link to="/" className="flex items-center gap-3">
            {/* ✅ 로고 더 크게 */}
            <img
              src="/logo/logo.png"
              alt="GA"
              className="h-14 w-14 object-contain select-none pointer-events-none"
              draggable={false}
            />
            <div className="leading-tight">
              <div className="text-[11px] font-semibold text-slate-300">
                {t("header:brand.subtitle", { defaultValue: "제스처 컨트롤 매니저" })}
              </div>
              <div className="text-lg font-black text-white tracking-tight">
                {t("header:brand.title", { defaultValue: "Gesture OS Manager" })}
              </div>
            </div>
          </Link>
        </div>

        {/* ✅ 모바일/태블릿에선 왼쪽도 그냥 로고+텍스트 보이게 */}
        <div className="flex lg:hidden items-center px-5">
          <Link to="/" className="flex items-center gap-3">
            <img
              src="/logo/logo.png"
              alt="GA"
              className="h-12 w-12 object-contain select-none pointer-events-none"
              draggable={false}
            />
            <div className="text-base font-black text-white tracking-tight">
              Gesture OS Manager
            </div>
          </Link>
        </div>

        {/* ✅ 오른쪽 영역 */}
        <div className="flex flex-1 items-center justify-end gap-4 px-6 lg:px-10">
          {/* 언어 드롭다운 */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setIsLangOpen((v) => !v)}
              className="flex items-center gap-2 rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 px-4 py-2 text-xs text-[var(--muted)] hover:text-white transition-all"
            >
              <svg
                className="w-4 h-4 text-[var(--accent)]"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"
                />
              </svg>
              {currentLang}
              <svg
                className={`w-4 h-4 transition-transform ${isLangOpen ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2.5}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>

            {isLangOpen && (
              <div className="absolute top-full right-0 mt-2 w-40 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-2 shadow-[0_18px_40px_rgba(6,12,26,0.55)] z-50">
                {LANGUAGES.map((lang) => (
                  <button
                    key={lang.code}
                    type="button"
                    onClick={() => selectLang(lang.code)}
                    className="w-full rounded-xl px-3 py-2 text-left text-xs text-[var(--muted)] hover:text-white hover:bg-[rgba(59,130,246,0.15)] transition-all"
                  >
                    {lang.name}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="h-7 w-[1px] bg-[var(--border)]" />

          {!isAuthed ? (
            <div className="flex items-center gap-2">
              <Link
                to="/login"
                className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/60 px-4 py-2 text-xs text-[var(--muted)] hover:text-white transition-all"
              >
                {t("header:auth.login", { defaultValue: "로그인" })}
              </Link>

              <Link
                to="/join"
                className="rounded-2xl bg-[var(--accent)] px-4 py-2 text-xs font-black text-white shadow-[0_10px_30px_rgba(59,130,246,0.35)] hover:bg-[var(--accent-strong)] transition-all"
              >
                {t("header:auth.join", { defaultValue: "회원가입" })}
              </Link>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <Link
                to="/mypage"
                className="flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--surface)]/70 px-3 py-1.5 text-xs text-[var(--muted)] hover:text-white transition-all"
              >
                <div className="h-7 w-7 rounded-full bg-[var(--accent)]/20 text-[var(--accent)] flex items-center justify-center">
                  <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <span>{user?.nickname || user?.name}</span>
              </Link>

              <button
                type="button"
                onClick={onLogout}
                className="rounded-2xl border border-[var(--border)] bg-[var(--surface)]/60 px-4 py-2 text-xs text-[var(--muted)] hover:text-white transition-all"
              >
                {t("header:auth.logout", { defaultValue: "로그아웃" })}
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
