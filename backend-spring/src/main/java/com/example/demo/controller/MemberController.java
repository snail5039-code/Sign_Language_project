package com.example.demo.controller;

import java.util.Map;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.dto.Member;
import com.example.demo.service.MemberService;

import jakarta.servlet.http.HttpSession;

@CrossOrigin(origins = "http://localhost:5173", allowCredentials = "true")
@RestController
@RequestMapping("/api/members")
public class MemberController {

	private MemberService memberService;
	
	public MemberController(MemberService memberService) {
		this.memberService =memberService; 
	}
	
	// 회원가입
    @PostMapping("/join")
    public Map<String, Object> join(@RequestBody Member member) {

        memberService.join(member);

        return Map.of("message", "회원가입 완료");
    }

    // 로그인 (세션 저장)
    @PostMapping("/login")
    public Map<String, Object> login(@RequestBody Map<String, String> body, HttpSession session) {
      String loginId = body.get("loginId");
      String loginPw = body.get("loginPw");

      Member m = memberService.login(loginId, loginPw);
      session.setAttribute("loginedMemberId", m.getId());

      return Map.of("message", "로그인 성공", "memberId", m.getId());
    }

    // 로그아웃
    @PostMapping("/logout")
    public Map<String, Object> logout(HttpSession session) {
        session.invalidate();
        return Map.of("message", "로그아웃");
      }
    
    
    @GetMapping("/me")
    public Map<String, Object> me(HttpSession session) {
      Object id = session.getAttribute("loginedMemberId");
      return Map.of("logined", id != null, "memberId", id);
    }
    
  }