export default function SocialLoginButtons() {
  const disabled = true; // 백 OAuth 연결되면 false로 바꾸고 redirect 넣기

  const onClick = (provider) => {
    // 백에서 OAuth2 붙이면 보통 이렇게 연결:
    // window.location.href = `/oauth2/authorization/${provider}`;
    alert(`${provider} 소셜 로그인은 백엔드 OAuth2 붙이면 연결할 거야!`);
  };

  return (
    <div className="space-y-2">
      <p className="text-sm text-gray-500">소셜 계정으로 계속하기</p>

      <button
        type="button"
        disabled={disabled}
        onClick={() => onClick("google")}
        className="w-full border rounded-xl py-3 bg-white hover:bg-gray-50 disabled:opacity-60"
      >
        Google로 계속
      </button>

      <button
        type="button"
        disabled={disabled}
        onClick={() => onClick("kakao")}
        className="w-full border rounded-xl py-3 bg-white hover:bg-gray-50 disabled:opacity-60"
      >
        Kakao로 계속
      </button>

      <button
        type="button"
        disabled={disabled}
        onClick={() => onClick("naver")}
        className="w-full border rounded-xl py-3 bg-white hover:bg-gray-50 disabled:opacity-60"
      >
        Naver로 계속
      </button>

      <p className="text-xs text-gray-400">
        (현재는 UI 자리만. 백 OAuth2 붙으면 버튼이 로그인으로 연결됨)
      </p>
    </div>
  );
}
