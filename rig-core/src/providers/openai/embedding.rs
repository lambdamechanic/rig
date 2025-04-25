use super::{ApiErrorResponse, ApiResponse, Client, Usage};
use crate::embeddings;
use crate::embeddings::EmbeddingError;
use reqwest::{header::HeaderValue, StatusCode};
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;

// ================================================================
// OpenAI Embedding API
// ================================================================
/// `text-embedding-3-large` embedding model
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-3-small` embedding model
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-ada-002` embedding model
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

impl From<ApiErrorResponse> for EmbeddingError {
    fn from(err: ApiErrorResponse) -> Self {
        EmbeddingError::ProviderError(err.message)
    }
}

impl From<ApiResponse<EmbeddingResponse>> for Result<EmbeddingResponse, EmbeddingError> {
    fn from(value: ApiResponse<EmbeddingResponse>) -> Self {
        match value {
            ApiResponse::Ok(response) => Ok(response),
            ApiResponse::Err(err) => Err(EmbeddingError::ProviderError(err.message)),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: usize,
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: Client,
    pub model: String,
    ndims: usize,
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1024;

    fn ndims(&self) -> usize {
        self.ndims
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents_vec = documents.into_iter().collect::<Vec<_>>();
        let request_body = json!({
            "model": self.model,
            "input": documents_vec,
        });

        loop {
            let response = self
                .client
                .post("/embeddings")
                .json(&request_body)
                .send()
                .await?;

            match response.status() {
                StatusCode::OK => {
                    // Success case
                    match response.json::<ApiResponse<EmbeddingResponse>>().await? {
                        ApiResponse::Ok(response_data) => {
                            tracing::info!(target: "rig",
                                "OpenAI embedding token usage: {}",
                                response_data.usage
                            );

                            if response_data.data.len() != documents_vec.len() {
                                return Err(EmbeddingError::ResponseError(
                                    "Response data length does not match input length".into(),
                                ));
                            }

                            return Ok(response_data
                                .data
                                .into_iter()
                                .zip(documents_vec.into_iter()) // Use the original vec here
                                .map(|(embedding, document)| embeddings::Embedding {
                                    document,
                                    vec: embedding.embedding,
                                })
                                .collect());
                        }
                        ApiResponse::Err(err) => {
                            return Err(EmbeddingError::ProviderError(err.message))
                        }
                    }
                }
                StatusCode::TOO_MANY_REQUESTS => {
                    // Rate limit exceeded, extract retry duration and wait
                    let retry_after = response
                        .headers()
                        .get("x-ratelimit-reset-requests")
                        .or_else(|| response.headers().get("x-ratelimit-reset-tokens"))
                        .and_then(parse_ratelimit_duration);

                    if let Some(duration) = retry_after {
                        tracing::warn!(target: "rig",
                            "Rate limit hit for OpenAI embeddings. Retrying after {:?}",
                            duration
                        );
                        sleep(duration).await;
                        continue; // Retry the request
                    } else {
                        // Header not found or couldn't parse, return error
                        let error_text = response.text().await?;
                        tracing::error!(target: "rig",
                            "Rate limit hit for OpenAI embeddings, but couldn't parse retry duration. Response: {}",
                            error_text
                        );
                        return Err(EmbeddingError::ProviderError(format!(
                            "Rate limit hit, but no valid retry duration found in headers. Response: {}",
                            error_text
                        )));
                    }
                }
                status => {
                    // Other error status codes
                    let error_text = response.text().await?;
                    tracing::error!(target: "rig",
                        "OpenAI embedding request failed with status {}: {}",
                        status, error_text
                    );
                    return Err(EmbeddingError::ProviderError(format!(
                        "Request failed with status {}: {}",
                        status, error_text
                    )));
                }
            }
        }
    }
}

/// Parses OpenAI's rate limit duration string (e.g., "6m10s", "500ms", "1s") into a Duration.
fn parse_ratelimit_duration(header_value: &HeaderValue) -> Option<Duration> {
    header_value.to_str().ok().and_then(|s| {
        let mut total_duration = Duration::ZERO;
        let mut current_value = String::new();

        for c in s.chars() {
            if c.is_ascii_digit() {
                current_value.push(c);
            } else if c.is_alphabetic() {
                if let Ok(val) = current_value.parse::<u64>() {
                    match c {
                        'h' => total_duration += Duration::from_secs(val * 3600),
                        'm' => {
                            // Check if the next char is 's' for 'ms'
                            if s.chars().skip_while(|&x| x != 'm').nth(1) == Some('s') {
                                // Skip 's' processing later
                            } else {
                                total_duration += Duration::from_secs(val * 60);
                            }
                        }
                        's' => {
                            // Check if previous char was 'm' for 'ms'
                            if s.chars().rev().skip_while(|&x| x != 's').nth(1) == Some('m') {
                                total_duration += Duration::from_millis(val);
                            } else {
                                total_duration += Duration::from_secs(val);
                            }
                        }
                        _ => { /* ignore unknown units */ }
                    }
                    current_value.clear();
                } else {
                    // Failed to parse number part
                    return None;
                }
            }
        }

        if total_duration == Duration::ZERO {
            None // No valid units found or parsed
        } else {
            Some(total_duration)
        }
    })
}

impl EmbeddingModel {
    pub fn new(client: Client, model: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.to_string(),
            ndims,
        }
    }
}
