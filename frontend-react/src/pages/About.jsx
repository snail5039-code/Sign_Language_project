import { useMemo } from "react";
import { useTranslation } from "react-i18next";

function cn(...xs) {
  return xs.filter(Boolean).join(" ");
}

function IconWave(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M3 15c2.4-4.8 5.6-7.2 9.6-7.2 3.5 0 6.1 1.8 7.4 5.4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M4 18c2.1-3.3 4.8-5 8.2-5 2.8 0 5 1.1 6.8 3.3"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        opacity="0.7"
      />
    </svg>
  );
}

function IconStack(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M12 3 3.5 7.5 12 12l8.5-4.5L12 3Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M3.5 12 12 16.5 20.5 12"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
        opacity="0.85"
      />
      <path
        d="M3.5 16.5 12 21l8.5-4.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
        opacity="0.7"
      />
    </svg>
  );
}

function IconMonitor(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M4 5h16v11H4V5Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M9 20h6"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M12 16v4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function InfoCard({ icon, title, desc }) {
  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-lg border border-[var(--border)] bg-[var(--surface-soft)] p-2 text-[color:var(--text)]">
          {icon}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-[color:var(--text)]">{title}</div>
          <div className="mt-2 text-sm leading-relaxed text-[var(--muted)]">{desc}</div>
        </div>
      </div>
    </div>
  );
}

function Bullet({ idx, text }) {
  return (
    <div className="flex gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3">
      <div className="mt-0.5 h-6 w-6 shrink-0 rounded-md border border-[var(--border)] bg-[var(--surface)] text-[11px] font-semibold text-[color:var(--text)] flex items-center justify-center">
        {String(idx + 1).padStart(2, "0")}
      </div>
      <div className="text-sm leading-relaxed text-[color:var(--text)]">{text}</div>
    </div>
  );
}

export default function About() {
  const { t } = useTranslation("about");

  const itemsRaw = t("core.items", { returnObjects: true });
  const items = useMemo(
    () => (Array.isArray(itemsRaw) ? itemsRaw.filter(Boolean) : []),
    [itemsRaw]
  );

  return (
    <div className="relative space-y-6">
      {/* 배경 데코: 과하지 않게 */}
      <div className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
        <div
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.06) 1px, transparent 1px)",
            backgroundSize: "52px 52px",
          }}
        />
        <div className="absolute -top-40 right-[-160px] h-[420px] w-[420px] rounded-full bg-emerald-400/16 blur-[90px]" />
        <div className="absolute -bottom-48 left-[-160px] h-[460px] w-[460px] rounded-full bg-sky-400/10 blur-[110px]" />
      </div>

      {/* Header */}
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <div className="text-xs tracking-wide text-[var(--muted)]">{t("navTitle")}</div>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight text-[color:var(--text)]">
            {t("title")}
          </h1>
        </div>

        <span className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-1 text-xs text-[var(--muted)]">
          {t("badge")}
        </span>
      </header>

      {/* HERO (수어 문구 제거: t("core.desc") 아예 안 씀) */}
      <section className="relative overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-7">
        <div className="absolute inset-x-0 top-0 h-[2px] bg-gradient-to-r from-emerald-400/55 via-sky-400/35 to-transparent" />

        <div className="inline-flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--surface-soft)] px-3 py-1 text-xs text-[var(--muted)]">
          <IconWave className="h-4 w-4 text-emerald-300/90" />
          <span>제스처 · 모션 기반 컴퓨터 제어</span>
        </div>

        <p className="mt-4 max-w-3xl text-sm leading-relaxed text-[var(--muted)]">
          Gesture OS Manager는 카메라 입력에서 손 동작을 인식해, 마우스/키보드/프레젠테이션 등
          다양한 모드로 컴퓨터를 제어하는 데스크톱 솔루션입니다. 현재 상태(모드/트래킹/잠금)를
          화면에 명확히 표시해 실제 사용 중 혼선을 줄이고, 바로 익혀서 쓰는 흐름에 집중했습니다.
        </p>

        <div className="mt-5 flex flex-wrap gap-2">
          <span className="rounded-md border border-[var(--border)] bg-[var(--surface-soft)] px-3 py-1 text-xs text-[color:var(--text)]">
            실시간 인식
          </span>
          <span className="rounded-md border border-[var(--border)] bg-[var(--surface-soft)] px-3 py-1 text-xs text-[color:var(--text)]">
            모드 전환
          </span>
          <span className="rounded-md border border-[var(--border)] bg-[var(--surface-soft)] px-3 py-1 text-xs text-[color:var(--text)]">
            즉시 피드백
          </span>
          <span className="rounded-md border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200">
            현재 1.0 버전 사용 가능
          </span>
        </div>
      </section>

      {/* 주요 특징 (불필요한 “핵심기능 N개” 같은 통계 제거) */}
      <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-7">
        <div className="flex items-end justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-[color:var(--text)]">
              주요 기능
            </h2>
            <p className="mt-2 text-sm text-[var(--muted)]">
              사용 중 체감되는 포인트
            </p>
          </div>
        </div>

        <div className="mt-5 grid gap-3 lg:grid-cols-2">
          {(items.length ? items : ["모션 인식 기반 제어", "모드별 동작 매핑", "상태/피드백 표시"]).map(
            (text, idx) => (
              <Bullet key={idx} idx={idx} text={text} />
            )
          )}
        </div>
      </section>

      {/* 구성/사용 흐름 */}
      <section className="grid gap-6 lg:grid-cols-3">
        <InfoCard
          icon={<IconStack className="h-5 w-5" />}
          title="구성"
          desc="매니저(설정/가이드)와 에이전트(인식/입력/HUD)가 분리되어 동작합니다. 사용자는 매니저에서 모드와 옵션을 조정하고, 에이전트가 실제 입력을 처리합니다."
        />
        <InfoCard
          icon={<IconWave className="h-5 w-5" />}
          title="사용 흐름"
          desc="모드 선택 → 트래킹 확인 → 제스처 수행 → 결과 피드백 확인. 현재 상태가 화면에 표시되므로, ‘왜 안 되는지’를 추측하지 않아도 됩니다."
        />
        <InfoCard
          icon={<IconMonitor className="h-5 w-5" />}
          title={t("env.title")}
          desc={t("env.desc")}
        />
      </section>

      {/* 배포 상태 (준비중/목표/핵심 수치 제거) */}
      <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-7">
        <h3 className="text-sm font-semibold text-[color:var(--text)]">1.0 버전 </h3>
        <p className="mt-2 text-sm leading-relaxed text-[var(--muted)]">
          현재 버전은 1.0버전으로 마우수, 키보드, 프레젠테이션, 키보드 모드를 사용할 수 있습니다. 추가적으로 RUSH 게임 등이 있습니다.
        </p>

        <div className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3">
          <div className="text-[11px] tracking-wide text-[var(--muted)]">권장</div>
          <div className="mt-1 text-sm text-[color:var(--text)]">
            조명 안정/카메라 고정 환경에서 인식 안정성이 가장 좋습니다.
          </div>
        </div>
      </section>
    </div>
  );
}
