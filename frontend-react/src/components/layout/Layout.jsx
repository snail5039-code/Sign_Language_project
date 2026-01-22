import { NavLink, Outlet } from "react-router-dom";
import { useTranslation } from "react-i18next";
import AppHeader from "./AppHeader";
import ChatWidget from "../help/ChatWidget";
import { useAuth } from "../../auth/AuthProvider";

export default function Layout() {
  const { isAuthed } = useAuth();
  const { t } = useTranslation(["layout"]);

  const navItems = [
    { to: "/", key: "home" },
    { to: "/about", key: "about" },
    { to: "/board", key: "board" },
    { to: "/motionGuide", key: "motionGuide" },
    { to: "/download", key: "download" },
  ];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(30,58,138,0.35),_transparent_50%),radial-gradient(circle_at_bottom,_rgba(8,47,73,0.35),_transparent_55%),linear-gradient(180deg,_var(--bg)_0%,_var(--bg-deep)_100%)]">
      {/* ✅ 헤더를 최상단에 두고, 아래에 사이드바+본문 */}
      <AppHeader />

      <div className="flex min-h-[calc(100vh-72px)]">
        <aside className="hidden lg:flex w-72 flex-col border-r border-[var(--border)] bg-[var(--surface-soft)]/70 backdrop-blur-xl">
          {/* ✅ 로고 영역 삭제 (AppHeader로 이동) */}
          <nav className="flex-1 px-4 py-4 space-y-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-3 rounded-2xl px-4 py-3 transition-all ${
                    isActive
                      ? "bg-[var(--surface)] text-white shadow-[0_12px_28px_rgba(6,12,26,0.5)]"
                      : "text-[var(--muted)] hover:text-white hover:bg-[rgba(59,130,246,0.12)]"
                  }`
                }
              >
                <span className="h-2 w-2 rounded-full bg-[var(--accent)]/70"></span>
                <span className="text-sm">{t(`layout:nav.${item.key}`)}</span>
              </NavLink>
            ))}
          </nav>

          <div className="px-6 pb-6 pt-2 text-xs text-[var(--muted)]">
            {isAuthed ? t("layout:status.signedIn") : t("layout:status.guest")}
          </div>
        </aside>

        <div className="flex flex-1 flex-col">
          <main className="flex-1 px-6 py-6 lg:px-10">
            <Outlet />
          </main>
        </div>
      </div>

      <ChatWidget />
    </div>
  );
}
