//package com.example.demo.security;
//
//import java.net.URLEncoder;
//import java.nio.charset.StandardCharsets;
//
//import org.springframework.security.core.Authentication;
//import org.springframework.security.oauth2.core.user.OAuth2User;
//import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
//import org.springframework.stereotype.Component;
//
//import com.example.demo.dto.Member;
//import com.example.demo.service.MemberService;
//
//import jakarta.servlet.http.HttpServletRequest;
//import jakarta.servlet.http.HttpServletResponse;
//
//@Component
//public class OAuth2SuccessHandler implements AuthenticationSuccessHandler {
//	
//	private MemberService memberService;
//	private JwtTokenProvider jwtTokenProvider;
//	
//	public OAuth2SuccessHandler (MemberService memberService, JwtTokenProvider jwtTokenProvider) {
//		this.memberService = memberService;
//		this.jwtTokenProvider = jwtTokenProvider;
//	}
//	
//	@Override
//	public void onAuthenticationSuccess(HttpServletRequest request, HttpServletResponse response, Authentication authentication) {
//		try {
//			OAuth2User user = (OAuth2User) authentication.getPrincipal();
//			
//			String email = (String) user.getAttributes().get("email");
//			String name = (String) user.getAttributes().get("name");
//			String providerKey = (String) user.getAttributes().get("sub");
//			
//			Member m = memberService.upsertSocialUser("GOOGLE", email, name, providerKey);
//			
//			String token = jwtTokenProvider.createToken(m.getId(), m.getLoginId());
//			
//            String redirect = "http://localhost:5173/oauth2/success?token="
//            		+ URLEncoder.encode(token, StandardCharsets.UTF_8);
//            
//            response.sendRedirect(redirect);
//		} catch (Exception e) {
//			throw new RuntimeException(e);
//		}
//	}
//
//}
