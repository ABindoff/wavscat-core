//! The single error type.

use std::fmt;

/// An invalid argument or an operator that cannot be built.
///
/// Messages follow the R package's wording, so a user sees the same advice
/// whichever language they call from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl Error {
    pub(crate) fn new(msg: impl Into<String>) -> Self {
        Error(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub(crate) fn fail<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error::new(msg))
}
