export default function MotionGuide() {
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-sm text-[var(--muted)]">Motion Guide</div>
          <h1 className="text-2xl tracking-tight text-white">제스처 모션 가이드</h1>
        </div>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-[var(--border)] px-3 py-1 text-xs text-[var(--muted)]">
            LIVE
          </span>
          <span className="rounded-full bg-[var(--accent)]/20 px-3 py-1 text-xs text-[var(--accent)]">
            READY
          </span>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <section className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[0_18px_40px_rgba(6,12,26,0.45)]">
          <h2 className="text-lg text-white">동작 리스트</h2>
          <p className="mt-2 text-sm text-[var(--muted)]">
            현재 지원하는 제스처를 확인하고 모션별 설명을 참고하세요.
          </p>
          <div className="mt-6 space-y-3">
            {[
              { name: "Open Palm", desc: "정지 / 대기 상태" },
              { name: "Closed Fist", desc: "클릭 / 선택" },
              { name: "Swipe Left", desc: "이전 항목" },
              { name: "Swipe Right", desc: "다음 항목" },
              { name: "Pinch", desc: "확대 / 강조" },
            ].map((item) => (
              <div
                key={item.name}
                className="flex items-center justify-between rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3"
              >
                <div>
                  <div className="text-sm text-white">{item.name}</div>
                  <div className="text-xs text-[var(--muted)]">{item.desc}</div>
                </div>
                <span className="rounded-full bg-[var(--accent)]/15 px-3 py-1 text-[10px] text-[var(--accent)]">
                  ACTIVE
                </span>
              </div>
            ))}
          </div>
        </section>

        <aside className="space-y-6">
          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[0_18px_40px_rgba(6,12,26,0.45)]">
            <h3 className="text-sm text-white">캘리브레이션</h3>
            <p className="mt-2 text-xs text-[var(--muted)]">
              카메라와 손의 거리를 일정하게 유지해 주세요.
            </p>
            <button className="mt-4 w-full rounded-2xl bg-[var(--accent)] px-4 py-2 text-xs text-white">
              캘리브레이션 시작
            </button>
          </div>

          <div className="rounded-3xl border border-[var(--border)] bg-[var(--surface)] p-6">
            <h3 className="text-sm text-white">주의사항</h3>
            <ul className="mt-3 space-y-2 text-xs text-[var(--muted)]">
              <li>조명이 어두우면 인식률이 낮아집니다.</li>
              <li>손이 카메라 프레임을 벗어나지 않도록 유지하세요.</li>
              <li>모션은 천천히, 크게 수행하는 것이 좋습니다.</li>
            </ul>
          </div>
        </aside>
      </div>
    </div>
  );
}
