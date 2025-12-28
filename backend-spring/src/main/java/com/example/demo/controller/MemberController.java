package com.example.demo.controller;

import java.util.Map;

import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.dto.Member;
import com.example.demo.security.JwtTokenProvider;
import com.example.demo.service.MemberService;

import jakarta.servlet.http.HttpSession;

@CrossOrigin(origins = "http://localhost:5173", allowCredentials = "true")
@RestController
@RequestMapping("/api/members")
public class MemberController {

	private MemberService memberService;
	private JwtTokenProvider jwtTokenProvider;
	
	public MemberController(MemberService memberService, JwtTokenProvider jwtTokenProvider) {
		this.memberService = memberService; 
		this.jwtTokenProvider = jwtTokenProvider;
	}
	
	// 회원가입
    @PostMapping("/join")
    public Map<String, Object> join(@RequestBody Member member) {

        memberService.join(member);

        return Map.of("message", "회원가입 완료");
    }

    // 로그인
    @PostMapping("/login")
    public Map<String, Object> login(@RequestBody Map<String, String> body) {
      String loginId = body.get("loginId");
      String loginPw = body.get("loginPw");

      Member m = memberService.login(loginId, loginPw);
      
      String token = jwtTokenProvider.createToken(m.getId(), m.getLoginId());

      return Map.of("message", "로그인 성공", "accessToken", token, "memberId", m.getId());
    }

    // 로그아웃
    @PostMapping("/logout")
    public Map<String, Object> logout() {
        return Map.of("message", "로그아웃");
      }
    
    
    @GetMapping("/me")
    public Map<String, Object> me(Authentication authentication) {
    	
        if (authentication == null) {
            return Map.of("logined", false);
        }
        Integer memberId = (Integer) authentication.getPrincipal();
        return Map.of("logined", true, "memberId", memberId);
    }
    
  }