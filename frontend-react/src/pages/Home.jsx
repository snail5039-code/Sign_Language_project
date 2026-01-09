export default function Home() {
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-sm text-[var(--muted)]">System Overview</div>
          <h1 className="text-3xl tracking-tight text-white">Gesture Control Manager</h1>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-[var(--border)] px-3 py-1 text-xs text-[var(--muted)]">
            REST polling 500ms
          </span>
          <span className="rounded-full bg-[var(--accent)]/20 px-3 py-1 text-xs text-[var(--accent)]">
            CONNECTED
          </span>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1.2fr,1fr]">
        <section className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[0_18px_40px_rgba(6,12,26,0.45)]">
          <h2 className="text-lg text-white">Live Status</h2>
          <p className="mt-2 text-sm text-[var(--muted)]">
            실시간으로 모션 인식 상태와 연결 정보를 확인합니다.
          </p>
          <div className="mt-6 grid gap-4 sm:grid-cols-2">
            {[
              { label: "Connection", value: "Stable", tone: "text-[var(--success)]" },
              { label: "Active Gesture", value: "None", tone: "text-[var(--muted)]" },
              { label: "FPS", value: "30.2", tone: "text-white" },
              { label: "Mode", value: "Mouse", tone: "text-white" },
            ].map((item) => (
              <div key={item.label} className="rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3">
                <div className="text-xs text-[var(--muted)]">{item.label}</div>
                <div className={`mt-1 text-lg ${item.tone}`}>{item.value}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="space-y-6">
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6">
            <h3 className="text-sm text-white">Quick Actions</h3>
            <div className="mt-4 grid grid-cols-2 gap-3">
              {["Start", "Stop", "Preview", "Refresh"].map((label) => (
                <button
                  key={label}
                  className="rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-2 text-xs text-[var(--muted)] hover:text-white transition-all"
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6">
            <h3 className="text-sm text-white">최근 업데이트</h3>
            <ul className="mt-3 space-y-3 text-xs text-[var(--muted)]">
              <li>Motion Guide 페이지가 준비되었습니다.</li>
              <li>게시판 UI가 앱 스타일로 변경됩니다.</li>
              <li>로그인 페이지 디자인을 새롭게 구성했습니다.</li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}
