//package com.example.demo.security;
//
//import org.springframework.context.annotation.Bean;
//import org.springframework.context.annotation.Configuration;
//import org.springframework.http.HttpMethod;
//import org.springframework.security.config.Customizer;
//import org.springframework.security.config.annotation.web.builders.HttpSecurity;
//import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
//import org.springframework.security.config.http.SessionCreationPolicy;
//import org.springframework.security.web.SecurityFilterChain;
//import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
//
//@Configuration
//@EnableWebSecurity
//public class SecurityConfig {
//
//	private final OAuth2SuccessHandler oAuth2SuccessHandler;
//
//	public SecurityConfig(OAuth2SuccessHandler oAuth2SuccessHandler) {
//	    this.oAuth2SuccessHandler = oAuth2SuccessHandler;
//	}
//	
//	@Bean
//	SecurityFilterChain filterChain(HttpSecurity http, JwtAuthFilter jwtAuthFilter) throws Exception {
//		
//		return http.csrf(csrf -> csrf.disable()).cors(Customizer.withDefaults())
//				.sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
//				.authorizeHttpRequests(auth -> auth
//						.requestMatchers("/api/health").permitAll()
//						.requestMatchers("/api/members/join", "/api/members/login").permitAll()
//						.requestMatchers("/api/members/me").permitAll()
//						.requestMatchers("/oauth2/**", "/login/oauth2/**").permitAll()
//						.requestMatchers(HttpMethod.GET, "/api/boards/**").permitAll()
//						.anyRequest().authenticated()
//				).addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class)
//				.oauth2Login(o -> o.successHandler(oAuth2SuccessHandler)) // 소셜은 아래에서 successHandler로 커스텀 예정
//				.build();
//	}
//	
//}
