package com.example.demo.controller;

import org.springframework.http.HttpStatus;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import com.example.demo.dto.Member;
import com.example.demo.dto.MyPageData;
import com.example.demo.info.GoogleUserInfo;
import com.example.demo.info.KakaoUserInfo;
import com.example.demo.info.NaverUserInfo;
import com.example.demo.dao.RefreshTokenDao;
import com.example.demo.dto.Country;
import com.example.demo.service.MemberService;
import com.example.demo.social.OAuth2UserInfo;
import com.example.demo.token.JwtTokenProvider;

import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletResponse;

import java.util.Map;
import java.util.List;

@RestController
@RequestMapping("/api/members")
public class MemberController {

    private final MemberService memberService;
    private final JwtTokenProvider jwtTokenProvider;
    private final RefreshTokenDao refreshTokenDao;

    public MemberController(MemberService memberService, JwtTokenProvider jwtTokenProvider,
            RefreshTokenDao refreshTokenDao) {
        this.memberService = memberService;
        this.jwtTokenProvider = jwtTokenProvider;
        this.refreshTokenDao = refreshTokenDao;
    }

    @GetMapping("/countries")
    public List<Country> getCountries() {
        return memberService.countries();
    }

    @PostMapping("/join")
    public Map<String, Object> join(@RequestBody Member member) {
        this.memberService.join(member);
        return Map.of("message", "회원가입 완료");
    }

    @GetMapping("/checkNickname")
    public Map<String, Object> checkNickname(@RequestParam String nickname) {
        boolean exists = memberService.isNicknameTaken(nickname);
        return Map.of("result", exists ? "fail" : "success");
    }

    @GetMapping("/checkLoginId")
    public Map<String, Object> checkLoginId(@RequestParam String loginId) {
        boolean exists = memberService.isLoginIdTaken(loginId);
        return Map.of("result", exists ? "fail" : "success");
    }

    @PostMapping("/login")
    public Map<String, Object> login(@RequestBody Map<String, String> body, HttpServletResponse response) {
        String loginId = body.getOrDefault("loginId", "");
        String loginPw = body.getOrDefault("loginPw", "");

        if (loginId.isBlank() || loginPw.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "아이디와 비밀번호를 입력해주세요.");
        }

        Member member = memberService.login(loginId, loginPw);
        return generateTokens(member, response);
    }

    @GetMapping("/oauth2/login")
    public Map<String, Object> oauthLogin(@AuthenticationPrincipal OAuth2User principal,
            @RequestParam("provider") String provider,
            HttpServletResponse response) {
        if (principal == null)
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED);

        Map<String, Object> attributes = principal.getAttributes();
        OAuth2UserInfo info = switch (provider.toLowerCase()) {
            case "google" -> new GoogleUserInfo(attributes);
            case "kakao" -> new KakaoUserInfo(attributes);
            case "naver" -> {
                Object responseObj = attributes.get("response");
                if (responseObj instanceof Map) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> responseMap = (Map<String, Object>) responseObj;
                    yield new NaverUserInfo(responseMap);
                }
                throw new IllegalStateException("Invalid Naver response format");
            }
            default -> throw new IllegalArgumentException("Unsupported provider");
        };

        Member m = memberService.upsertSocialUser(
                provider.toUpperCase(),
                info.getEmail(),
                info.getName(),
                info.getProviderKey());

        return generateTokens(m, response);
    }

    @GetMapping("/me")
    public Map<String, Object> getCurrentMember(@AuthenticationPrincipal Object principal) {
        if (principal == null || principal.toString().equals("anonymousUser")) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "로그인 필요");
        }

        if (principal instanceof Integer) {
            Member member = memberService.findById((Integer) principal);
            return Map.of("logined", true, "user", member);
        }

        return Map.of("logined", true, "user", principal);
    }

    @PostMapping("/logout")
    public Map<String, Object> logout(HttpServletResponse response) {
        Cookie cookie = new Cookie("refreshToken", "");
        cookie.setHttpOnly(true);
        cookie.setPath("/");
        cookie.setMaxAge(0);
        response.addCookie(cookie);
        return Map.of("message", "로그아웃 완료");
    }

    @PostMapping("/findLoginId")
    public Map<String, Object> findLoginId(@RequestBody Map<String, String> body) {
        String name = body.get("name");
        String email = body.get("email");
        if (name == null || email == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "이름과 이메일을 입력해주세요.");
        }

        memberService.findLoginIdByNameAndEmail(name.trim(), email.trim());
        return Map.of("message", "입력하신 이메일로 아이디를 전송했습니다.");
    }

    @PostMapping("/findLoginPw")
    public Map<String, Object> findLoginPw(@RequestBody Map<String, String> body) {
        String loginId = body.get("loginId");
        String email = body.get("email");
        if (loginId == null || email == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "아이디와 이메일을 입력해주세요.");
        }

        memberService.resetPasswordWithEmail(loginId.trim(), email.trim());
        return Map.of("message", "입력하신 이메일로 임시 비밀번호를 전송했습니다.");
    }

    @PostMapping("/sendVerificationCode")
    public Map<String, Object> sendVerificationCode(@RequestBody Map<String, String> body) {
        String email = body.get("email");
        if (email == null || email.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "이메일을 입력해주세요.");
        }
        memberService.sendVerificationCode(email.trim());
        return Map.of("message", "인증 코드가 발송되었습니다.");
    }

    @PostMapping("/verifyCode")
    public Map<String, Object> verifyCode(@RequestBody Map<String, String> body) {
        String email = body.get("email");
        String code = body.get("code");
        if (email == null || code == null || email.isBlank() || code.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "이메일과 인증 코드를 입력해주세요.");
        }
        memberService.verifyCode(email.trim(), code.trim());
        return Map.of("message", "이메일 인증이 완료되었습니다.");
    }

    private Map<String, Object> generateTokens(Member member, HttpServletResponse response) {
        String role = member.getRole();
        if (role != null && !role.startsWith("ROLE_")) {
            role = "ROLE_" + role;
        }

        String accessToken = jwtTokenProvider.createAccessToken(member.getId(), member.getEmail(), role);
        String refreshToken = jwtTokenProvider.createRefreshToken(member.getId());

        refreshTokenDao.upsert(member.getId(), refreshToken);

        Cookie cookie = new Cookie("refreshToken", refreshToken);
        cookie.setHttpOnly(true);
        cookie.setPath("/");
        cookie.setMaxAge(60 * 60 * 24 * 7); // 7일
        response.addCookie(cookie);

        return Map.of(
                "accessToken", accessToken,
                "memberId", member.getId(),
                "name", member.getName(),
                "role", member.getRole());
    }

    @GetMapping("/mypage")
    public MyPageData getMyPage(Authentication auth) {
        if (auth == null) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "로그인 필요");
        }
        Integer memberId = (Integer) auth.getPrincipal();
        return memberService.getMyPageData(memberId);
    }

    @PutMapping("/modify/{id}")
    public Map<String, Object> modify(@PathVariable int id, @RequestBody Member member, Authentication auth) {
        System.out.println("[MemberController] modify id=" + id + ", member=" + member);
        if (auth == null) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "로그인이 필요합니다.");
        }

        Integer loginMemberId = (Integer) auth.getPrincipal();
        if (loginMemberId != id) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "권한이 없습니다.");
        }

        member.setId(id);
        this.memberService.memberModify(member, id);

        return Map.of("message", "수정완료");
    }

    @DeleteMapping("/delete/{id}")
    public Map<String, Object> delete(@PathVariable int id, Authentication auth) {
        if (auth == null) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "로그인이 필요합니다.");
        }
        Integer loginMemberId = (Integer) auth.getPrincipal();
        if (loginMemberId != id) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "권한이 없습니다.");
        }
        memberService.memberDelete(id);
        return Map.of("message", "삭제완료");
    }

}