export default function About() {
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-sm text-[var(--muted)]">About</div>
          <h1 className="text-3xl tracking-tight text-white">프로그램 소개</h1>
        </div>
        <span className="rounded-full border border-[var(--border)] px-3 py-1 text-xs text-[var(--muted)]">
          SLT Project
        </span>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1.2fr,1fr]">
        <section className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-8">
          <h2 className="text-lg text-white">핵심 기능</h2>
          <p className="mt-3 text-sm text-[var(--muted)]">
            수어 기반 제스처로 컴퓨터를 제어하고, 실시간 상호작용을 돕는
            통합 솔루션을 제공합니다.
          </p>
          <div className="mt-6 space-y-3 text-sm text-slate-200">
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3">
              모션 인식 기반 제어
            </div>
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3">
              사용자 친화적인 가이드
            </div>
            <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3">
              보드/피드백 시스템
            </div>
          </div>
        </section>

        <aside className="space-y-6">
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-8">
            <h3 className="text-sm text-white">지원 환경</h3>
            <p className="mt-2 text-xs text-[var(--muted)]">
              Windows 환경에서 실행되는 데스크톱 앱(.exe)을 제공합니다.
            </p>
          </div>
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-8">
            <h3 className="text-sm text-white">진행 상태</h3>
            <p className="mt-2 text-xs text-[var(--muted)]">
              현재 데스크톱 버전 배포 준비 중입니다.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
