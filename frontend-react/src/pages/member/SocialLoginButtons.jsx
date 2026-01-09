export default function SocialLoginButtons() {
  const goToLogin = (provider) => {
    // 소셜 로그인 제공자 URL로 리디렉션
    window.location.href = `/oauth2/authorization/${provider}`;
  };

  return (
    <div className="space-y-3">
      <button
        type="button"
        onClick={() => goToLogin("google")}
        className="flex w-full items-center justify-between rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3 text-xs text-white hover:border-[var(--accent)] transition-all"
      >
        <span>Google</span>
        <span className="text-[10px] text-[var(--muted)]">Continue</span>
      </button>

      <button
        type="button"
        onClick={() => goToLogin("kakao")}
        className="flex w-full items-center justify-between rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3 text-xs text-white hover:border-[var(--accent)] transition-all"
      >
        <span>Kakao</span>
        <span className="text-[10px] text-[var(--muted)]">Continue</span>
      </button>

      <button
        type="button"
        onClick={() => goToLogin("naver")}
        className="flex w-full items-center justify-between rounded-2xl border border-[var(--border)] bg-[var(--surface-soft)] px-4 py-3 text-xs text-white hover:border-[var(--accent)] transition-all"
      >
        <span>Naver</span>
        <span className="text-[10px] text-[var(--muted)]">Continue</span>
      </button>
    </div>
  );
}
