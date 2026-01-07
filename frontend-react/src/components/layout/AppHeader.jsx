import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../auth/AuthProvider";
import useLangMenu from "../../shared/useLangMenu";

const BOARD_MENU = [
  { key: "notice" },
  { key: "free" },
  { key: "qna" },
  { key: "error" },
];

export default function AppHeader() {
  const { user, isAuthed, logout } = useAuth();
  const nav = useNavigate();

  const { t } = useTranslation(["header", "board"]);
  const { LANGUAGES, isLangOpen, setIsLangOpen, currentLang, selectLang } =
    useLangMenu();

  const onLogout = async () => {
    await logout();
    nav("/");
  };

  return (
    <header className="sticky top-0 bg-white/90 backdrop-blur-xl border-b border-slate-200/50 z-50 shadow-sm">
      <div className="max-w-[1400px] mx-auto px-6 h-20 flex items-center justify-between">
        <div className="flex items-center gap-12">
          <Link
            to="/"
            className="text-2xl font-black text-indigo-600 tracking-tighter hover:opacity-80 transition-opacity"
          >
            {t("header:brand")}
          </Link>

          <nav className="hidden lg:flex items-center gap-1">
            <Link
              to="/encyclopedia"
              className="px-5 py-2 text-sm font-black text-slate-700 hover:text-indigo-600 hover:bg-indigo-50 rounded-2xl transition-all"
            >
              {t("header:nav.encyclopedia")}
            </Link>

            <Link
              to="/camera"
              className="px-5 py-2 text-sm font-black text-slate-700 hover:text-indigo-600 hover:bg-indigo-50 rounded-2xl transition-all"
            >
              {t("header:nav.translate")}
            </Link>

            <Link
              to="/call"
              className="px-5 py-2 text-sm font-black text-slate-700 hover:text-indigo-600 hover:bg-indigo-50 rounded-2xl transition-all"
            >
              {t("header:nav.call")}
            </Link>

            <div className="relative group">
              <Link
                to="/board"
                className="px-5 py-2 text-sm font-black text-slate-700 hover:text-indigo-600 hover:bg-indigo-50 rounded-2xl transition-all flex items-center gap-1"
              >
                {t("header:nav.board")}
                <svg
                  className="w-4 h-4 transition-transform group-hover:rotate-180"
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
              </Link>

              <div className="absolute top-full left-0 mt-2 w-48 bg-white rounded-3xl shadow-2xl border border-slate-100 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all transform origin-top scale-95 group-hover:scale-100 p-2 z-50">
                {BOARD_MENU.map((item) => (
                  <Link
                    key={item.key}
                    to={`/board?type=${item.key}`}
                    className="block px-4 py-3 text-sm font-bold text-slate-600 hover:bg-indigo-50 hover:text-indigo-600 rounded-2xl transition-all"
                  >
                    {t(`board:menu.${item.key}`)}
                  </Link>
                ))}
              </div>
            </div>
          </nav>
        </div>

        <div className="flex items-center gap-4">
          {/* 언어 드롭다운 */}
          <div className="relative">
            <button
              type="button"
              onClick={() => {
                console.log("[LangMenu] toggle click", { isLangOpen });
                setIsLangOpen((v) => !v);
              }}
              className="flex items-center gap-2 px-4 py-2 text-sm font-black text-slate-700 hover:bg-slate-100 rounded-2xl transition-all"
            >
              <svg
                className="w-4 h-4 text-indigo-500"
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
                className={`w-4 h-4 transition-transform ${
                  isLangOpen ? "rotate-180" : ""
                }`}
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
              <div className="absolute top-full right-0 mt-2 w-40 bg-white rounded-3xl shadow-2xl border border-slate-100 p-2 z-50">
                {LANGUAGES.map((lang) => (
                  <button
                    key={lang.code}
                    type="button"
                    onClick={() => {
                      console.log("[LangMenu] select click", lang.code);
                      selectLang(lang.code);
                    }}
                    className="w-full text-left px-4 py-3 text-sm font-bold text-slate-600 hover:bg-indigo-50 hover:text-indigo-600 rounded-2xl transition-all"
                  >
                    {lang.name}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="h-6 w-[1px] bg-slate-200 mx-2"></div>

          {!isAuthed ? (
            <div className="flex items-center gap-2">
              <Link
                to="/login"
                className="px-5 py-2.5 text-sm font-black text-slate-700 hover:bg-slate-100 rounded-2xl transition-all"
              >
                {t("header:auth.login")}
              </Link>
              <Link
                to="/join"
                className="px-6 py-2.5 rounded-2xl bg-indigo-600 text-white text-sm font-black shadow-lg shadow-indigo-100 hover:bg-indigo-700 hover:-translate-y-0.5 transition-all"
              >
                {t("header:auth.join")}
              </Link>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <Link
                to="/mypage"
                className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 hover:bg-slate-100 rounded-full border border-slate-200 transition-all"
              >
                <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-indigo-600"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <span className="text-xs font-black text-slate-700">
                  {user?.nickname || user?.name}
                </span>
              </Link>

              <button
                type="button"
                onClick={onLogout}
                className="px-5 py-2.5 rounded-2xl bg-slate-900 text-white text-sm font-black hover:bg-slate-800 transition-all shadow-md"
              >
                {t("header:auth.logout")}
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
