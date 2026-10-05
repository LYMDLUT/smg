//! The policy catalog: names, parameters, construction.
//!
//! Parameters are YAML (JSON is YAML) text, for example
//! `--selection-policy ramjet --selection-policy-params '{alpha: 2.0, basis: absolute}'`.
//! Unknown parameters are rejected, so a typo cannot silently fall back to a default.

use super::{default, dualmap, dynamo, llm_d, policy::WorkerSelectionPolicy, ramjet};

/// The policy used when none is configured: the pre-policy cache-aware decision.
pub const DEFAULT_POLICY: &str = default::POLICY_NAME;

pub const POLICY_NAMES: &[&str] = &[
    default::POLICY_NAME,
    dynamo::POLICY_NAME,
    llm_d::OPTIMIZED_BASELINE,
    llm_d::PRECISE_PREFIX,
    llm_d::STICKY_UNTIL_SATURATED,
    ramjet::POLICY_NAME,
    dualmap::POLICY_NAME,
];

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("unknown selection policy '{0}'; known: {known}", known = POLICY_NAMES.join(", "))]
    Unknown(String),
    #[error("selection policy '{name}' parameters: {message}")]
    Parameters { name: String, message: String },
}

fn parse<T: Default + serde::de::DeserializeOwned>(
    name: &str,
    params: Option<&str>,
) -> Result<T, CatalogError> {
    match params.map(str::trim).filter(|p| !p.is_empty()) {
        None => Ok(T::default()),
        Some(text) => serde_yaml::from_str(text).map_err(|e| CatalogError::Parameters {
            name: name.to_string(),
            message: e.to_string(),
        }),
    }
}

fn validated<T>(
    name: &str,
    params: T,
    check: impl Fn(&T) -> Result<(), String>,
) -> Result<T, CatalogError> {
    check(&params).map_err(|message| CatalogError::Parameters {
        name: name.to_string(),
        message,
    })?;
    Ok(params)
}

fn llm_d_policy(
    name: &'static str,
    preset: llm_d::LlmDParams,
    params: Option<&str>,
) -> Result<WorkerSelectionPolicy, CatalogError> {
    let patch = parse::<llm_d::LlmDParamsPatch>(name, params)?;
    let p = validated(name, preset.apply(&patch), |p| p.validate())?;
    Ok(llm_d::policy(name, p))
}

/// The default policy at the given cache-aware temperature; it takes no parameters and cannot
/// fail to build.
pub fn default_policy(selection_temperature: f32) -> WorkerSelectionPolicy {
    default::policy(selection_temperature)
}

/// Build a policy by name. `selection_temperature` is the cache-aware temperature the default
/// policy keeps using; the ported policies carry their own temperature in their parameters.
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
        dynamo::POLICY_NAME => {
            let p = validated(
                name,
                parse::<dynamo::DynamoDefaultParams>(name, params)?,
                |p| p.validate(),
            )?;
            Ok(dynamo::policy(p))
        }
        llm_d::OPTIMIZED_BASELINE => llm_d_policy(
            llm_d::OPTIMIZED_BASELINE,
            llm_d::LlmDParams::optimized_baseline(),
            params,
        ),
        llm_d::PRECISE_PREFIX => llm_d_policy(
            llm_d::PRECISE_PREFIX,
            llm_d::LlmDParams::precise_prefix(),
            params,
        ),
        llm_d::STICKY_UNTIL_SATURATED => llm_d_policy(
            llm_d::STICKY_UNTIL_SATURATED,
            llm_d::LlmDParams::sticky_until_saturated(),
            params,
        ),
        ramjet::POLICY_NAME => {
            let p = validated(name, parse::<ramjet::RamjetParams>(name, params)?, |p| {
                p.validate()
            })?;
            Ok(ramjet::policy(p))
        }
        dualmap::POLICY_NAME => {
            let p = validated(name, parse::<dualmap::DualMapParams>(name, params)?, |p| {
                p.validate()
            })?;
            Ok(dualmap::policy(p))
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
        assert!(build("ramjet", Some("{alpha: 2.5, basis: absolute}"), 0.0).is_ok());
        assert!(matches!(
            build("ramjet", Some("{alphaa: 2.5}"), 0.0),
            Err(CatalogError::Parameters { .. })
        ));
        assert!(matches!(
            build(
                "llm-d-sticky-until-saturated",
                Some("{affinity_threshold: 1.5}"),
                0.0
            ),
            Err(CatalogError::Parameters { .. })
        ));
        assert!(build("llm-d-precise-prefix", Some("{prefix_weight: 1}"), 0.0).is_ok());
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
