//! The policy catalog: names, parameters, construction.
//!
//! Parameters are YAML (JSON is YAML) text. Unknown parameters are rejected, so a typo cannot
//! silently fall back to a default.

use super::{default, policy::WorkerSelectionPolicy};

/// The policy used when none is configured: the pre-policy cache-aware decision.
pub const DEFAULT_POLICY: &str = default::POLICY_NAME;

pub const POLICY_NAMES: &[&str] = &[default::POLICY_NAME];

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("unknown selection policy '{0}'; known: {known}", known = POLICY_NAMES.join(", "))]
    Unknown(String),
    #[error("selection policy '{name}' parameters: {message}")]
    Parameters { name: String, message: String },
}

/// The default policy at the given cache-aware temperature; it takes no parameters and cannot
/// fail to build.
pub fn default_policy(selection_temperature: f32) -> WorkerSelectionPolicy {
    default::policy(selection_temperature)
}

/// Build a policy by name. `selection_temperature` is the cache-aware temperature the default
/// policy keeps using.
pub fn build(
    name: &str,
    params: Option<&str>,
    selection_temperature: f32,
) -> Result<WorkerSelectionPolicy, CatalogError> {
    match name {
        default::POLICY_NAME => {
            if params.is_some_and(|p| !p.trim().is_empty()) {
                return Err(CatalogError::Parameters {
                    name: name.to_string(),
                    message: "takes no parameters; tune it with the cache-aware flags".into(),
                });
            }
            Ok(default::policy(selection_temperature))
        }
        other => Err(CatalogError::Unknown(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_builds_with_defaults() {
        for name in POLICY_NAMES {
            let policy = build(name, None, 0.0).expect("default parameters build");
            assert_eq!(policy.name(), *name);
        }
    }

    #[test]
    fn parameters_are_parsed_and_checked() {
        assert!(matches!(
            build("nope", None, 0.0),
            Err(CatalogError::Unknown(_))
        ));
        assert!(matches!(
            build(DEFAULT_POLICY, Some("{x: 1}"), 0.0),
            Err(CatalogError::Parameters { .. })
        ));
    }
}
