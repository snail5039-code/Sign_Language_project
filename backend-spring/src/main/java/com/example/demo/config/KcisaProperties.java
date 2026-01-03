package com.example.demo.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "kcisa.integrated")
public class KcisaProperties {
	private String baseUrl;
	private String serviceKey;
	private int numOfRows = 10;
	
	public String getBaseUrl() {
		return baseUrl;
	}
	
	public void setBaseUrl(String baseUrl) {
		this.baseUrl = baseUrl;
	}
	
	public String getServiceKey() {
		return serviceKey;
	}
	
	public void setServiceKey(String serviceKey) {
		this.serviceKey = serviceKey;
	}
	
	public int getNumOfRows() {
		return numOfRows;
	}
	
	public void setNumOfRows(int numOfRows) {
		this.numOfRows = numOfRows;
	}
}
