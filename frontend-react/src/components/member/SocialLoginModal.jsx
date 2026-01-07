import React from "react";

export default function SocialLoginModal() {
    const goToLogin = (provider) => {
        window.location.href = `/oauth2/authorization/${provider}`;
    };

    return (
        <div className="space-y-4">
            <button
                onClick={() => goToLogin("google")}
                className="w-full flex items-center justify-center gap-3 py-4 bg-white border border-slate-200 rounded-2xl font-bold hover:bg-slate-50 transition-all shadow-sm"
            >
                <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" alt="Google" className="w-5 h-5" />
                Login with Google
            </button>

            <button
                onClick={() => goToLogin("naver")}
                className="w-full flex items-center justify-center gap-3 py-4 bg-[#03C75A] text-white rounded-2xl font-bold hover:opacity-90 transition-all shadow-sm"
            >
                <span className="font-black text-xl">N</span>
                네이버 로그인
            </button>

            <button
                onClick={() => goToLogin("kakao")}
                className="w-full flex items-center justify-center gap-3 py-4 bg-[#FEE500] text-[#3c1e1e] rounded-2xl font-bold hover:opacity-90 transition-all shadow-sm"
            >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 3c-4.97 0-9 3.185-9 7.115 0 2.557 1.707 4.8 4.337 6.107l-1.107 4.054c-.05.18.054.37.23.425.05.015.1.02.15.02.13 0 .25-.075.305-.195l4.643-3.102c.145.013.293.02.442.02 4.97 0 9-3.185 9-7.115S16.97 3 12 3z" />
                </svg>
                카카오 로그인
            </button>
        </div>
    );
}
