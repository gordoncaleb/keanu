impl serde::Serialize for ApproxEq {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.epsilon != 0. {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.ApproxEq", len)?;
        if self.epsilon != 0. {
            struct_ser.serialize_field("epsilon", &self.epsilon)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ApproxEq {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "epsilon",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Epsilon,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "epsilon" => Ok(GeneratedField::Epsilon),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ApproxEq;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.ApproxEq")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ApproxEq, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut epsilon__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Epsilon => {
                            if epsilon__.is_some() {
                                return Err(serde::de::Error::duplicate_field("epsilon"));
                            }
                            epsilon__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(ApproxEq {
                    epsilon: epsilon__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.ApproxEq", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ArgOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "ARG_OP_UNSPECIFIED",
            Self::ArgMax => "ARG_OP_ARG_MAX",
            Self::ArgMin => "ARG_OP_ARG_MIN",
            Self::NanArgMax => "ARG_OP_NAN_ARG_MAX",
            Self::NanArgMin => "ARG_OP_NAN_ARG_MIN",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for ArgOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ARG_OP_UNSPECIFIED",
            "ARG_OP_ARG_MAX",
            "ARG_OP_ARG_MIN",
            "ARG_OP_NAN_ARG_MAX",
            "ARG_OP_NAN_ARG_MIN",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = ArgOp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "ARG_OP_UNSPECIFIED" => Ok(ArgOp::Unspecified),
                    "ARG_OP_ARG_MAX" => Ok(ArgOp::ArgMax),
                    "ARG_OP_ARG_MIN" => Ok(ArgOp::ArgMin),
                    "ARG_OP_NAN_ARG_MAX" => Ok(ArgOp::NanArgMax),
                    "ARG_OP_NAN_ARG_MIN" => Ok(ArgOp::NanArgMin),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for ArgReduce {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind != 0 {
            len += 1;
        }
        if self.axis.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.ArgReduce", len)?;
        if self.kind != 0 {
            let v = ArgOp::try_from(self.kind)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.kind)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.axis.as_ref() {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&v).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ArgReduce {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "axis",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Axis,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "kind" => Ok(GeneratedField::Kind),
                            "axis" => Ok(GeneratedField::Axis),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ArgReduce;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.ArgReduce")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ArgReduce, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut axis__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = Some(map_.next_value::<ArgOp>()? as i32);
                        }
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                map_.next_value::<::std::option::Option<::pbjson::private::NumberDeserialize<_>>>()?.map(|x| x.0)
                            ;
                        }
                    }
                }
                Ok(ArgReduce {
                    kind: kind__.unwrap_or_default(),
                    axis: axis__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.ArgReduce", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Assert {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.message.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Assert", len)?;
        if let Some(v) = self.message.as_ref() {
            struct_ser.serialize_field("message", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Assert {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "message",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Message,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "message" => Ok(GeneratedField::Message),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Assert;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Assert")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Assert, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut message__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Message => {
                            if message__.is_some() {
                                return Err(serde::de::Error::duplicate_field("message"));
                            }
                            message__ = map_.next_value()?;
                        }
                    }
                }
                Ok(Assert {
                    message: message__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.Assert", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Axes {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Axes", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Axes {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Values,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "values" => Ok(GeneratedField::Values),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Axes;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Axes")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Axes, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(Axes {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Axes", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for BinaryOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "BINARY_OP_UNSPECIFIED",
            Self::Add => "BINARY_OP_ADD",
            Self::Sub => "BINARY_OP_SUB",
            Self::Mul => "BINARY_OP_MUL",
            Self::Div => "BINARY_OP_DIV",
            Self::Pow => "BINARY_OP_POW",
            Self::Max => "BINARY_OP_MAX",
            Self::Min => "BINARY_OP_MIN",
            Self::Rem => "BINARY_OP_REM",
            Self::Atan2 => "BINARY_OP_ATAN2",
            Self::LogAddExp => "BINARY_OP_LOG_ADD_EXP",
            Self::LogAddExp2 => "BINARY_OP_LOG_ADD_EXP2",
            Self::SafeLnTimes => "BINARY_OP_SAFE_LN_TIMES",
            Self::LogBeta => "BINARY_OP_LOG_BETA",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for BinaryOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "BINARY_OP_UNSPECIFIED",
            "BINARY_OP_ADD",
            "BINARY_OP_SUB",
            "BINARY_OP_MUL",
            "BINARY_OP_DIV",
            "BINARY_OP_POW",
            "BINARY_OP_MAX",
            "BINARY_OP_MIN",
            "BINARY_OP_REM",
            "BINARY_OP_ATAN2",
            "BINARY_OP_LOG_ADD_EXP",
            "BINARY_OP_LOG_ADD_EXP2",
            "BINARY_OP_SAFE_LN_TIMES",
            "BINARY_OP_LOG_BETA",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = BinaryOp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "BINARY_OP_UNSPECIFIED" => Ok(BinaryOp::Unspecified),
                    "BINARY_OP_ADD" => Ok(BinaryOp::Add),
                    "BINARY_OP_SUB" => Ok(BinaryOp::Sub),
                    "BINARY_OP_MUL" => Ok(BinaryOp::Mul),
                    "BINARY_OP_DIV" => Ok(BinaryOp::Div),
                    "BINARY_OP_POW" => Ok(BinaryOp::Pow),
                    "BINARY_OP_MAX" => Ok(BinaryOp::Max),
                    "BINARY_OP_MIN" => Ok(BinaryOp::Min),
                    "BINARY_OP_REM" => Ok(BinaryOp::Rem),
                    "BINARY_OP_ATAN2" => Ok(BinaryOp::Atan2),
                    "BINARY_OP_LOG_ADD_EXP" => Ok(BinaryOp::LogAddExp),
                    "BINARY_OP_LOG_ADD_EXP2" => Ok(BinaryOp::LogAddExp2),
                    "BINARY_OP_SAFE_LN_TIMES" => Ok(BinaryOp::SafeLnTimes),
                    "BINARY_OP_LOG_BETA" => Ok(BinaryOp::LogBeta),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for BoolValues {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.BoolValues", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for BoolValues {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Values,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "values" => Ok(GeneratedField::Values),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = BoolValues;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.BoolValues")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<BoolValues, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(BoolValues {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.BoolValues", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Broadcast {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.shape.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Broadcast", len)?;
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Broadcast {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shape",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Shape,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "shape" => Ok(GeneratedField::Shape),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Broadcast;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Broadcast")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Broadcast, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut shape__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Shape => {
                            if shape__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(Broadcast {
                    shape: shape__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Broadcast", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CompareOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "COMPARE_OP_UNSPECIFIED",
            Self::Gt => "COMPARE_OP_GT",
            Self::Ge => "COMPARE_OP_GE",
            Self::Lt => "COMPARE_OP_LT",
            Self::Le => "COMPARE_OP_LE",
            Self::Eq => "COMPARE_OP_EQ",
            Self::Ne => "COMPARE_OP_NE",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for CompareOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "COMPARE_OP_UNSPECIFIED",
            "COMPARE_OP_GT",
            "COMPARE_OP_GE",
            "COMPARE_OP_LT",
            "COMPARE_OP_LE",
            "COMPARE_OP_EQ",
            "COMPARE_OP_NE",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = CompareOp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "COMPARE_OP_UNSPECIFIED" => Ok(CompareOp::Unspecified),
                    "COMPARE_OP_GT" => Ok(CompareOp::Gt),
                    "COMPARE_OP_GE" => Ok(CompareOp::Ge),
                    "COMPARE_OP_LT" => Ok(CompareOp::Lt),
                    "COMPARE_OP_LE" => Ok(CompareOp::Le),
                    "COMPARE_OP_EQ" => Ok(CompareOp::Eq),
                    "COMPARE_OP_NE" => Ok(CompareOp::Ne),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for Concat {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.axis != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Concat", len)?;
        if self.axis != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&self.axis).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Concat {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "axis",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Axis,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "axis" => Ok(GeneratedField::Axis),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Concat;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Concat")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Concat, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut axis__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Concat {
                    axis: axis__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Concat", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Cpt {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.conditions != 0 {
            len += 1;
        }
        if !self.keys.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Cpt", len)?;
        if self.conditions != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("conditions", ToString::to_string(&self.conditions).as_str())?;
        }
        if !self.keys.is_empty() {
            struct_ser.serialize_field("keys", &self.keys)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Cpt {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "conditions",
            "keys",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Conditions,
            Keys,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "conditions" => Ok(GeneratedField::Conditions),
                            "keys" => Ok(GeneratedField::Keys),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Cpt;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Cpt")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Cpt, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut conditions__ = None;
                let mut keys__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Conditions => {
                            if conditions__.is_some() {
                                return Err(serde::de::Error::duplicate_field("conditions"));
                            }
                            conditions__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Keys => {
                            if keys__.is_some() {
                                return Err(serde::de::Error::duplicate_field("keys"));
                            }
                            keys__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(Cpt {
                    conditions: conditions__.unwrap_or_default(),
                    keys: keys__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Cpt", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CptKey {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.CptKey", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for CptKey {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Values,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "values" => Ok(GeneratedField::Values),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = CptKey;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.CptKey")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<CptKey, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(CptKey {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.CptKey", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Cumulative {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind != 0 {
            len += 1;
        }
        if self.axis != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Cumulative", len)?;
        if self.kind != 0 {
            let v = CumulativeOp::try_from(self.kind)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.kind)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if self.axis != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&self.axis).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Cumulative {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "axis",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Axis,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "kind" => Ok(GeneratedField::Kind),
                            "axis" => Ok(GeneratedField::Axis),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Cumulative;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Cumulative")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Cumulative, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut axis__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = Some(map_.next_value::<CumulativeOp>()? as i32);
                        }
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Cumulative {
                    kind: kind__.unwrap_or_default(),
                    axis: axis__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Cumulative", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for CumulativeOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "CUMULATIVE_OP_UNSPECIFIED",
            Self::Sum => "CUMULATIVE_OP_SUM",
            Self::Product => "CUMULATIVE_OP_PRODUCT",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for CumulativeOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "CUMULATIVE_OP_UNSPECIFIED",
            "CUMULATIVE_OP_SUM",
            "CUMULATIVE_OP_PRODUCT",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = CumulativeOp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "CUMULATIVE_OP_UNSPECIFIED" => Ok(CumulativeOp::Unspecified),
                    "CUMULATIVE_OP_SUM" => Ok(CumulativeOp::Sum),
                    "CUMULATIVE_OP_PRODUCT" => Ok(CumulativeOp::Product),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for DType {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "D_TYPE_UNSPECIFIED",
            Self::F64 => "D_TYPE_F64",
            Self::I64 => "D_TYPE_I64",
            Self::Bool => "D_TYPE_BOOL",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for DType {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "D_TYPE_UNSPECIFIED",
            "D_TYPE_F64",
            "D_TYPE_I64",
            "D_TYPE_BOOL",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = DType;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "D_TYPE_UNSPECIFIED" => Ok(DType::Unspecified),
                    "D_TYPE_F64" => Ok(DType::F64),
                    "D_TYPE_I64" => Ok(DType::I64),
                    "D_TYPE_BOOL" => Ok(DType::Bool),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for Distribution {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind != 0 {
            len += 1;
        }
        if !self.shape.is_empty() {
            len += 1;
        }
        if self.kde.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Distribution", len)?;
        if self.kind != 0 {
            let v = DistributionKind::try_from(self.kind)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.kind)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if let Some(v) = self.kde.as_ref() {
            struct_ser.serialize_field("kde", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Distribution {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "shape",
            "kde",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Shape,
            Kde,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "kind" => Ok(GeneratedField::Kind),
                            "shape" => Ok(GeneratedField::Shape),
                            "kde" => Ok(GeneratedField::Kde),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Distribution;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Distribution")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Distribution, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut shape__ = None;
                let mut kde__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = Some(map_.next_value::<DistributionKind>()? as i32);
                        }
                        GeneratedField::Shape => {
                            if shape__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Kde => {
                            if kde__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kde"));
                            }
                            kde__ = map_.next_value()?;
                        }
                    }
                }
                Ok(Distribution {
                    kind: kind__.unwrap_or_default(),
                    shape: shape__.unwrap_or_default(),
                    kde: kde__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.Distribution", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for DistributionKind {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "DISTRIBUTION_KIND_UNSPECIFIED",
            Self::Gaussian => "DISTRIBUTION_KIND_GAUSSIAN",
            Self::HalfGaussian => "DISTRIBUTION_KIND_HALF_GAUSSIAN",
            Self::Cauchy => "DISTRIBUTION_KIND_CAUCHY",
            Self::HalfCauchy => "DISTRIBUTION_KIND_HALF_CAUCHY",
            Self::Beta => "DISTRIBUTION_KIND_BETA",
            Self::ChiSquared => "DISTRIBUTION_KIND_CHI_SQUARED",
            Self::Dirichlet => "DISTRIBUTION_KIND_DIRICHLET",
            Self::Exponential => "DISTRIBUTION_KIND_EXPONENTIAL",
            Self::Gamma => "DISTRIBUTION_KIND_GAMMA",
            Self::InverseGamma => "DISTRIBUTION_KIND_INVERSE_GAMMA",
            Self::Laplace => "DISTRIBUTION_KIND_LAPLACE",
            Self::Logistic => "DISTRIBUTION_KIND_LOGISTIC",
            Self::LogNormal => "DISTRIBUTION_KIND_LOG_NORMAL",
            Self::MultivariateGaussian => "DISTRIBUTION_KIND_MULTIVARIATE_GAUSSIAN",
            Self::Pareto => "DISTRIBUTION_KIND_PARETO",
            Self::StudentT => "DISTRIBUTION_KIND_STUDENT_T",
            Self::Triangular => "DISTRIBUTION_KIND_TRIANGULAR",
            Self::Uniform => "DISTRIBUTION_KIND_UNIFORM",
            Self::Kde => "DISTRIBUTION_KIND_KDE",
            Self::Bernoulli => "DISTRIBUTION_KIND_BERNOULLI",
            Self::Binomial => "DISTRIBUTION_KIND_BINOMIAL",
            Self::Categorical => "DISTRIBUTION_KIND_CATEGORICAL",
            Self::Geometric => "DISTRIBUTION_KIND_GEOMETRIC",
            Self::Multinomial => "DISTRIBUTION_KIND_MULTINOMIAL",
            Self::Poisson => "DISTRIBUTION_KIND_POISSON",
            Self::UniformInt => "DISTRIBUTION_KIND_UNIFORM_INT",
            Self::LkjCholesky => "DISTRIBUTION_KIND_LKJ_CHOLESKY",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for DistributionKind {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "DISTRIBUTION_KIND_UNSPECIFIED",
            "DISTRIBUTION_KIND_GAUSSIAN",
            "DISTRIBUTION_KIND_HALF_GAUSSIAN",
            "DISTRIBUTION_KIND_CAUCHY",
            "DISTRIBUTION_KIND_HALF_CAUCHY",
            "DISTRIBUTION_KIND_BETA",
            "DISTRIBUTION_KIND_CHI_SQUARED",
            "DISTRIBUTION_KIND_DIRICHLET",
            "DISTRIBUTION_KIND_EXPONENTIAL",
            "DISTRIBUTION_KIND_GAMMA",
            "DISTRIBUTION_KIND_INVERSE_GAMMA",
            "DISTRIBUTION_KIND_LAPLACE",
            "DISTRIBUTION_KIND_LOGISTIC",
            "DISTRIBUTION_KIND_LOG_NORMAL",
            "DISTRIBUTION_KIND_MULTIVARIATE_GAUSSIAN",
            "DISTRIBUTION_KIND_PARETO",
            "DISTRIBUTION_KIND_STUDENT_T",
            "DISTRIBUTION_KIND_TRIANGULAR",
            "DISTRIBUTION_KIND_UNIFORM",
            "DISTRIBUTION_KIND_KDE",
            "DISTRIBUTION_KIND_BERNOULLI",
            "DISTRIBUTION_KIND_BINOMIAL",
            "DISTRIBUTION_KIND_CATEGORICAL",
            "DISTRIBUTION_KIND_GEOMETRIC",
            "DISTRIBUTION_KIND_MULTINOMIAL",
            "DISTRIBUTION_KIND_POISSON",
            "DISTRIBUTION_KIND_UNIFORM_INT",
            "DISTRIBUTION_KIND_LKJ_CHOLESKY",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = DistributionKind;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "DISTRIBUTION_KIND_UNSPECIFIED" => Ok(DistributionKind::Unspecified),
                    "DISTRIBUTION_KIND_GAUSSIAN" => Ok(DistributionKind::Gaussian),
                    "DISTRIBUTION_KIND_HALF_GAUSSIAN" => Ok(DistributionKind::HalfGaussian),
                    "DISTRIBUTION_KIND_CAUCHY" => Ok(DistributionKind::Cauchy),
                    "DISTRIBUTION_KIND_HALF_CAUCHY" => Ok(DistributionKind::HalfCauchy),
                    "DISTRIBUTION_KIND_BETA" => Ok(DistributionKind::Beta),
                    "DISTRIBUTION_KIND_CHI_SQUARED" => Ok(DistributionKind::ChiSquared),
                    "DISTRIBUTION_KIND_DIRICHLET" => Ok(DistributionKind::Dirichlet),
                    "DISTRIBUTION_KIND_EXPONENTIAL" => Ok(DistributionKind::Exponential),
                    "DISTRIBUTION_KIND_GAMMA" => Ok(DistributionKind::Gamma),
                    "DISTRIBUTION_KIND_INVERSE_GAMMA" => Ok(DistributionKind::InverseGamma),
                    "DISTRIBUTION_KIND_LAPLACE" => Ok(DistributionKind::Laplace),
                    "DISTRIBUTION_KIND_LOGISTIC" => Ok(DistributionKind::Logistic),
                    "DISTRIBUTION_KIND_LOG_NORMAL" => Ok(DistributionKind::LogNormal),
                    "DISTRIBUTION_KIND_MULTIVARIATE_GAUSSIAN" => Ok(DistributionKind::MultivariateGaussian),
                    "DISTRIBUTION_KIND_PARETO" => Ok(DistributionKind::Pareto),
                    "DISTRIBUTION_KIND_STUDENT_T" => Ok(DistributionKind::StudentT),
                    "DISTRIBUTION_KIND_TRIANGULAR" => Ok(DistributionKind::Triangular),
                    "DISTRIBUTION_KIND_UNIFORM" => Ok(DistributionKind::Uniform),
                    "DISTRIBUTION_KIND_KDE" => Ok(DistributionKind::Kde),
                    "DISTRIBUTION_KIND_BERNOULLI" => Ok(DistributionKind::Bernoulli),
                    "DISTRIBUTION_KIND_BINOMIAL" => Ok(DistributionKind::Binomial),
                    "DISTRIBUTION_KIND_CATEGORICAL" => Ok(DistributionKind::Categorical),
                    "DISTRIBUTION_KIND_GEOMETRIC" => Ok(DistributionKind::Geometric),
                    "DISTRIBUTION_KIND_MULTINOMIAL" => Ok(DistributionKind::Multinomial),
                    "DISTRIBUTION_KIND_POISSON" => Ok(DistributionKind::Poisson),
                    "DISTRIBUTION_KIND_UNIFORM_INT" => Ok(DistributionKind::UniformInt),
                    "DISTRIBUTION_KIND_LKJ_CHOLESKY" => Ok(DistributionKind::LkjCholesky),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for DoubleValues {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.DoubleValues", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for DoubleValues {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Values,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "values" => Ok(GeneratedField::Values),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = DoubleValues;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.DoubleValues")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<DoubleValues, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(DoubleValues {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.DoubleValues", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Empty {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let len = 0;
        let struct_ser = serializer.serialize_struct("keanu.Empty", len)?;
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Empty {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                            Err(serde::de::Error::unknown_field(value, FIELDS))
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Empty;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Empty")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Empty, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                while map_.next_key::<GeneratedField>()?.is_some() {
                    let _ = map_.next_value::<serde::de::IgnoredAny>()?;
                }
                Ok(Empty {
                })
            }
        }
        deserializer.deserialize_struct("keanu.Empty", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for FillTriangular {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.upper {
            len += 1;
        }
        if self.lower {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.FillTriangular", len)?;
        if self.upper {
            struct_ser.serialize_field("upper", &self.upper)?;
        }
        if self.lower {
            struct_ser.serialize_field("lower", &self.lower)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for FillTriangular {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "upper",
            "lower",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Upper,
            Lower,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "upper" => Ok(GeneratedField::Upper),
                            "lower" => Ok(GeneratedField::Lower),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = FillTriangular;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.FillTriangular")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<FillTriangular, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut upper__ = None;
                let mut lower__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Upper => {
                            if upper__.is_some() {
                                return Err(serde::de::Error::duplicate_field("upper"));
                            }
                            upper__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Lower => {
                            if lower__.is_some() {
                                return Err(serde::de::Error::duplicate_field("lower"));
                            }
                            lower__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(FillTriangular {
                    upper: upper__.unwrap_or_default(),
                    lower: lower__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.FillTriangular", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Gather {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.axis != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Gather", len)?;
        if self.axis != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&self.axis).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Gather {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "axis",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Axis,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "axis" => Ok(GeneratedField::Axis),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Gather;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Gather")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Gather, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut axis__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Gather {
                    axis: axis__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Gather", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Graph {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.nodes.is_empty() {
            len += 1;
        }
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Graph", len)?;
        if !self.nodes.is_empty() {
            struct_ser.serialize_field("nodes", &self.nodes)?;
        }
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Graph {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "nodes",
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Nodes,
            Values,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "nodes" => Ok(GeneratedField::Nodes),
                            "values" => Ok(GeneratedField::Values),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Graph;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Graph")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Graph, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut nodes__ = None;
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Nodes => {
                            if nodes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("nodes"));
                            }
                            nodes__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(Graph {
                    nodes: nodes__.unwrap_or_default(),
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Graph", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for IntValues {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.IntValues", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for IntValues {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Values,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "values" => Ok(GeneratedField::Values),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = IntValues;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.IntValues")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<IntValues, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(IntValues {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.IntValues", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Kde {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.samples.is_empty() {
            len += 1;
        }
        if self.bandwidth != 0. {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Kde", len)?;
        if !self.samples.is_empty() {
            struct_ser.serialize_field("samples", &self.samples)?;
        }
        if self.bandwidth != 0. {
            struct_ser.serialize_field("bandwidth", &self.bandwidth)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Kde {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "samples",
            "bandwidth",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Samples,
            Bandwidth,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "samples" => Ok(GeneratedField::Samples),
                            "bandwidth" => Ok(GeneratedField::Bandwidth),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Kde;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Kde")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Kde, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut samples__ = None;
                let mut bandwidth__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Samples => {
                            if samples__.is_some() {
                                return Err(serde::de::Error::duplicate_field("samples"));
                            }
                            samples__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Bandwidth => {
                            if bandwidth__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bandwidth"));
                            }
                            bandwidth__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Kde {
                    samples: samples__.unwrap_or_default(),
                    bandwidth: bandwidth__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Kde", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for LogSoftmax {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.axis != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.LogSoftmax", len)?;
        if self.axis != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&self.axis).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for LogSoftmax {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "axis",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Axis,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "axis" => Ok(GeneratedField::Axis),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = LogSoftmax;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.LogSoftmax")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<LogSoftmax, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut axis__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(LogSoftmax {
                    axis: axis__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.LogSoftmax", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for LogicOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "LOGIC_OP_UNSPECIFIED",
            Self::And => "LOGIC_OP_AND",
            Self::Or => "LOGIC_OP_OR",
            Self::Xor => "LOGIC_OP_XOR",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for LogicOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "LOGIC_OP_UNSPECIFIED",
            "LOGIC_OP_AND",
            "LOGIC_OP_OR",
            "LOGIC_OP_XOR",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = LogicOp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "LOGIC_OP_UNSPECIFIED" => Ok(LogicOp::Unspecified),
                    "LOGIC_OP_AND" => Ok(LogicOp::And),
                    "LOGIC_OP_OR" => Ok(LogicOp::Or),
                    "LOGIC_OP_XOR" => Ok(LogicOp::Xor),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for MatMul {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.transpose_left {
            len += 1;
        }
        if self.transpose_right {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.MatMul", len)?;
        if self.transpose_left {
            struct_ser.serialize_field("transposeLeft", &self.transpose_left)?;
        }
        if self.transpose_right {
            struct_ser.serialize_field("transposeRight", &self.transpose_right)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for MatMul {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "transpose_left",
            "transposeLeft",
            "transpose_right",
            "transposeRight",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            TransposeLeft,
            TransposeRight,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "transposeLeft" | "transpose_left" => Ok(GeneratedField::TransposeLeft),
                            "transposeRight" | "transpose_right" => Ok(GeneratedField::TransposeRight),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = MatMul;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.MatMul")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<MatMul, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut transpose_left__ = None;
                let mut transpose_right__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::TransposeLeft => {
                            if transpose_left__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transposeLeft"));
                            }
                            transpose_left__ = Some(map_.next_value()?);
                        }
                        GeneratedField::TransposeRight => {
                            if transpose_right__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transposeRight"));
                            }
                            transpose_right__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(MatMul {
                    transpose_left: transpose_left__.unwrap_or_default(),
                    transpose_right: transpose_right__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.MatMul", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Model {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.format_version != 0 {
            len += 1;
        }
        if self.graph.is_some() {
            len += 1;
        }
        if !self.metadata.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Model", len)?;
        if self.format_version != 0 {
            struct_ser.serialize_field("formatVersion", &self.format_version)?;
        }
        if let Some(v) = self.graph.as_ref() {
            struct_ser.serialize_field("graph", v)?;
        }
        if !self.metadata.is_empty() {
            struct_ser.serialize_field("metadata", &self.metadata)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Model {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "format_version",
            "formatVersion",
            "graph",
            "metadata",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            FormatVersion,
            Graph,
            Metadata,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "formatVersion" | "format_version" => Ok(GeneratedField::FormatVersion),
                            "graph" => Ok(GeneratedField::Graph),
                            "metadata" => Ok(GeneratedField::Metadata),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Model;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Model")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Model, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut format_version__ = None;
                let mut graph__ = None;
                let mut metadata__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::FormatVersion => {
                            if format_version__.is_some() {
                                return Err(serde::de::Error::duplicate_field("formatVersion"));
                            }
                            format_version__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Graph => {
                            if graph__.is_some() {
                                return Err(serde::de::Error::duplicate_field("graph"));
                            }
                            graph__ = map_.next_value()?;
                        }
                        GeneratedField::Metadata => {
                            if metadata__.is_some() {
                                return Err(serde::de::Error::duplicate_field("metadata"));
                            }
                            metadata__ = Some(
                                map_.next_value::<std::collections::BTreeMap<_, _>>()?
                            );
                        }
                    }
                }
                Ok(Model {
                    format_version: format_version__.unwrap_or_default(),
                    graph: graph__,
                    metadata: metadata__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Model", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Node {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.parents.is_empty() {
            len += 1;
        }
        if !self.label.is_empty() {
            len += 1;
        }
        if self.op.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Node", len)?;
        if !self.parents.is_empty() {
            struct_ser.serialize_field("parents", &self.parents)?;
        }
        if !self.label.is_empty() {
            struct_ser.serialize_field("label", &self.label)?;
        }
        if let Some(v) = self.op.as_ref() {
            match v {
                node::Op::Constant(v) => {
                    struct_ser.serialize_field("constant", v)?;
                }
                node::Op::Unary(v) => {
                    let v = UnaryOp::try_from(*v)
                        .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
                    struct_ser.serialize_field("unary", &v)?;
                }
                node::Op::Binary(v) => {
                    let v = BinaryOp::try_from(*v)
                        .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
                    struct_ser.serialize_field("binary", &v)?;
                }
                node::Op::Compare(v) => {
                    let v = CompareOp::try_from(*v)
                        .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
                    struct_ser.serialize_field("compare", &v)?;
                }
                node::Op::Mask(v) => {
                    let v = CompareOp::try_from(*v)
                        .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
                    struct_ser.serialize_field("mask", &v)?;
                }
                node::Op::ApproxEq(v) => {
                    struct_ser.serialize_field("approxEq", v)?;
                }
                node::Op::Logic(v) => {
                    let v = LogicOp::try_from(*v)
                        .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
                    struct_ser.serialize_field("logic", &v)?;
                }
                node::Op::Not(v) => {
                    struct_ser.serialize_field("not", v)?;
                }
                node::Op::All(v) => {
                    struct_ser.serialize_field("all", v)?;
                }
                node::Op::Any(v) => {
                    struct_ser.serialize_field("any", v)?;
                }
                node::Op::Predicate(v) => {
                    let v = Predicate::try_from(*v)
                        .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
                    struct_ser.serialize_field("predicate", &v)?;
                }
                node::Op::Reduce(v) => {
                    struct_ser.serialize_field("reduce", v)?;
                }
                node::Op::Standardize(v) => {
                    struct_ser.serialize_field("standardize", v)?;
                }
                node::Op::Cumulative(v) => {
                    struct_ser.serialize_field("cumulative", v)?;
                }
                node::Op::ArgReduce(v) => {
                    struct_ser.serialize_field("argReduce", v)?;
                }
                node::Op::Reshape(v) => {
                    struct_ser.serialize_field("reshape", v)?;
                }
                node::Op::Permute(v) => {
                    struct_ser.serialize_field("permute", v)?;
                }
                node::Op::Transpose(v) => {
                    struct_ser.serialize_field("transpose", v)?;
                }
                node::Op::Broadcast(v) => {
                    struct_ser.serialize_field("broadcast", v)?;
                }
                node::Op::Slice(v) => {
                    struct_ser.serialize_field("slice", v)?;
                }
                node::Op::Concat(v) => {
                    struct_ser.serialize_field("concat", v)?;
                }
                node::Op::Take(v) => {
                    struct_ser.serialize_field("take", v)?;
                }
                node::Op::Diag(v) => {
                    struct_ser.serialize_field("diag", v)?;
                }
                node::Op::DiagPart(v) => {
                    struct_ser.serialize_field("diagPart", v)?;
                }
                node::Op::Tril(v) => {
                    struct_ser.serialize_field("tril", v)?;
                }
                node::Op::Triu(v) => {
                    struct_ser.serialize_field("triu", v)?;
                }
                node::Op::FillTriangular(v) => {
                    struct_ser.serialize_field("fillTriangular", v)?;
                }
                node::Op::TrianglePart(v) => {
                    struct_ser.serialize_field("trianglePart", v)?;
                }
                node::Op::Matmul(v) => {
                    struct_ser.serialize_field("matmul", v)?;
                }
                node::Op::TensorMultiply(v) => {
                    struct_ser.serialize_field("tensorMultiply", v)?;
                }
                node::Op::Inverse(v) => {
                    struct_ser.serialize_field("inverse", v)?;
                }
                node::Op::Determinant(v) => {
                    struct_ser.serialize_field("determinant", v)?;
                }
                node::Op::Cholesky(v) => {
                    struct_ser.serialize_field("cholesky", v)?;
                }
                node::Op::CholeskyInverse(v) => {
                    struct_ser.serialize_field("choleskyInverse", v)?;
                }
                node::Op::Cast(v) => {
                    let v = DType::try_from(*v)
                        .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", *v)))?;
                    struct_ser.serialize_field("cast", &v)?;
                }
                node::Op::Where(v) => {
                    struct_ser.serialize_field("where", v)?;
                }
                node::Op::Multiplexer(v) => {
                    struct_ser.serialize_field("multiplexer", v)?;
                }
                node::Op::Cpt(v) => {
                    struct_ser.serialize_field("cpt", v)?;
                }
                node::Op::SetWithMask(v) => {
                    struct_ser.serialize_field("setWithMask", v)?;
                }
                node::Op::ReplaceNan(v) => {
                    struct_ser.serialize_field("replaceNan", v)?;
                }
                node::Op::Proxy(v) => {
                    struct_ser.serialize_field("proxy", v)?;
                }
                node::Op::Assert(v) => {
                    struct_ser.serialize_field("assert", v)?;
                }
                node::Op::Distribution(v) => {
                    struct_ser.serialize_field("distribution", v)?;
                }
                node::Op::Gather(v) => {
                    struct_ser.serialize_field("gather", v)?;
                }
                node::Op::ScatterAdd(v) => {
                    struct_ser.serialize_field("scatterAdd", v)?;
                }
                node::Op::Stack(v) => {
                    struct_ser.serialize_field("stack", v)?;
                }
                node::Op::Softmax(v) => {
                    struct_ser.serialize_field("softmax", v)?;
                }
                node::Op::LogSoftmax(v) => {
                    struct_ser.serialize_field("logSoftmax", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Node {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "parents",
            "label",
            "constant",
            "unary",
            "binary",
            "compare",
            "mask",
            "approx_eq",
            "approxEq",
            "logic",
            "not",
            "all",
            "any",
            "predicate",
            "reduce",
            "standardize",
            "cumulative",
            "arg_reduce",
            "argReduce",
            "reshape",
            "permute",
            "transpose",
            "broadcast",
            "slice",
            "concat",
            "take",
            "diag",
            "diag_part",
            "diagPart",
            "tril",
            "triu",
            "fill_triangular",
            "fillTriangular",
            "triangle_part",
            "trianglePart",
            "matmul",
            "tensor_multiply",
            "tensorMultiply",
            "inverse",
            "determinant",
            "cholesky",
            "cholesky_inverse",
            "choleskyInverse",
            "cast",
            "where",
            "multiplexer",
            "cpt",
            "set_with_mask",
            "setWithMask",
            "replace_nan",
            "replaceNan",
            "proxy",
            "assert",
            "distribution",
            "gather",
            "scatter_add",
            "scatterAdd",
            "stack",
            "softmax",
            "log_softmax",
            "logSoftmax",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Parents,
            Label,
            Constant,
            Unary,
            Binary,
            Compare,
            Mask,
            ApproxEq,
            Logic,
            Not,
            All,
            Any,
            Predicate,
            Reduce,
            Standardize,
            Cumulative,
            ArgReduce,
            Reshape,
            Permute,
            Transpose,
            Broadcast,
            Slice,
            Concat,
            Take,
            Diag,
            DiagPart,
            Tril,
            Triu,
            FillTriangular,
            TrianglePart,
            Matmul,
            TensorMultiply,
            Inverse,
            Determinant,
            Cholesky,
            CholeskyInverse,
            Cast,
            Where,
            Multiplexer,
            Cpt,
            SetWithMask,
            ReplaceNan,
            Proxy,
            Assert,
            Distribution,
            Gather,
            ScatterAdd,
            Stack,
            Softmax,
            LogSoftmax,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "parents" => Ok(GeneratedField::Parents),
                            "label" => Ok(GeneratedField::Label),
                            "constant" => Ok(GeneratedField::Constant),
                            "unary" => Ok(GeneratedField::Unary),
                            "binary" => Ok(GeneratedField::Binary),
                            "compare" => Ok(GeneratedField::Compare),
                            "mask" => Ok(GeneratedField::Mask),
                            "approxEq" | "approx_eq" => Ok(GeneratedField::ApproxEq),
                            "logic" => Ok(GeneratedField::Logic),
                            "not" => Ok(GeneratedField::Not),
                            "all" => Ok(GeneratedField::All),
                            "any" => Ok(GeneratedField::Any),
                            "predicate" => Ok(GeneratedField::Predicate),
                            "reduce" => Ok(GeneratedField::Reduce),
                            "standardize" => Ok(GeneratedField::Standardize),
                            "cumulative" => Ok(GeneratedField::Cumulative),
                            "argReduce" | "arg_reduce" => Ok(GeneratedField::ArgReduce),
                            "reshape" => Ok(GeneratedField::Reshape),
                            "permute" => Ok(GeneratedField::Permute),
                            "transpose" => Ok(GeneratedField::Transpose),
                            "broadcast" => Ok(GeneratedField::Broadcast),
                            "slice" => Ok(GeneratedField::Slice),
                            "concat" => Ok(GeneratedField::Concat),
                            "take" => Ok(GeneratedField::Take),
                            "diag" => Ok(GeneratedField::Diag),
                            "diagPart" | "diag_part" => Ok(GeneratedField::DiagPart),
                            "tril" => Ok(GeneratedField::Tril),
                            "triu" => Ok(GeneratedField::Triu),
                            "fillTriangular" | "fill_triangular" => Ok(GeneratedField::FillTriangular),
                            "trianglePart" | "triangle_part" => Ok(GeneratedField::TrianglePart),
                            "matmul" => Ok(GeneratedField::Matmul),
                            "tensorMultiply" | "tensor_multiply" => Ok(GeneratedField::TensorMultiply),
                            "inverse" => Ok(GeneratedField::Inverse),
                            "determinant" => Ok(GeneratedField::Determinant),
                            "cholesky" => Ok(GeneratedField::Cholesky),
                            "choleskyInverse" | "cholesky_inverse" => Ok(GeneratedField::CholeskyInverse),
                            "cast" => Ok(GeneratedField::Cast),
                            "where" => Ok(GeneratedField::Where),
                            "multiplexer" => Ok(GeneratedField::Multiplexer),
                            "cpt" => Ok(GeneratedField::Cpt),
                            "setWithMask" | "set_with_mask" => Ok(GeneratedField::SetWithMask),
                            "replaceNan" | "replace_nan" => Ok(GeneratedField::ReplaceNan),
                            "proxy" => Ok(GeneratedField::Proxy),
                            "assert" => Ok(GeneratedField::Assert),
                            "distribution" => Ok(GeneratedField::Distribution),
                            "gather" => Ok(GeneratedField::Gather),
                            "scatterAdd" | "scatter_add" => Ok(GeneratedField::ScatterAdd),
                            "stack" => Ok(GeneratedField::Stack),
                            "softmax" => Ok(GeneratedField::Softmax),
                            "logSoftmax" | "log_softmax" => Ok(GeneratedField::LogSoftmax),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Node;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Node")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Node, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut parents__ = None;
                let mut label__ = None;
                let mut op__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Parents => {
                            if parents__.is_some() {
                                return Err(serde::de::Error::duplicate_field("parents"));
                            }
                            parents__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Label => {
                            if label__.is_some() {
                                return Err(serde::de::Error::duplicate_field("label"));
                            }
                            label__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Constant => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("constant"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Constant)
;
                        }
                        GeneratedField::Unary => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("unary"));
                            }
                            op__ = map_.next_value::<::std::option::Option<UnaryOp>>()?.map(|x| node::Op::Unary(x as i32));
                        }
                        GeneratedField::Binary => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("binary"));
                            }
                            op__ = map_.next_value::<::std::option::Option<BinaryOp>>()?.map(|x| node::Op::Binary(x as i32));
                        }
                        GeneratedField::Compare => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("compare"));
                            }
                            op__ = map_.next_value::<::std::option::Option<CompareOp>>()?.map(|x| node::Op::Compare(x as i32));
                        }
                        GeneratedField::Mask => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("mask"));
                            }
                            op__ = map_.next_value::<::std::option::Option<CompareOp>>()?.map(|x| node::Op::Mask(x as i32));
                        }
                        GeneratedField::ApproxEq => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("approxEq"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::ApproxEq)
;
                        }
                        GeneratedField::Logic => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("logic"));
                            }
                            op__ = map_.next_value::<::std::option::Option<LogicOp>>()?.map(|x| node::Op::Logic(x as i32));
                        }
                        GeneratedField::Not => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("not"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Not)
;
                        }
                        GeneratedField::All => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("all"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::All)
;
                        }
                        GeneratedField::Any => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("any"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Any)
;
                        }
                        GeneratedField::Predicate => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("predicate"));
                            }
                            op__ = map_.next_value::<::std::option::Option<Predicate>>()?.map(|x| node::Op::Predicate(x as i32));
                        }
                        GeneratedField::Reduce => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reduce"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Reduce)
;
                        }
                        GeneratedField::Standardize => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("standardize"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Standardize)
;
                        }
                        GeneratedField::Cumulative => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("cumulative"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Cumulative)
;
                        }
                        GeneratedField::ArgReduce => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("argReduce"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::ArgReduce)
;
                        }
                        GeneratedField::Reshape => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("reshape"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Reshape)
;
                        }
                        GeneratedField::Permute => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("permute"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Permute)
;
                        }
                        GeneratedField::Transpose => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transpose"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Transpose)
;
                        }
                        GeneratedField::Broadcast => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("broadcast"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Broadcast)
;
                        }
                        GeneratedField::Slice => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("slice"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Slice)
;
                        }
                        GeneratedField::Concat => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("concat"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Concat)
;
                        }
                        GeneratedField::Take => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("take"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Take)
;
                        }
                        GeneratedField::Diag => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("diag"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Diag)
;
                        }
                        GeneratedField::DiagPart => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("diagPart"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::DiagPart)
;
                        }
                        GeneratedField::Tril => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("tril"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Tril)
;
                        }
                        GeneratedField::Triu => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("triu"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Triu)
;
                        }
                        GeneratedField::FillTriangular => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("fillTriangular"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::FillTriangular)
;
                        }
                        GeneratedField::TrianglePart => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("trianglePart"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::TrianglePart)
;
                        }
                        GeneratedField::Matmul => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("matmul"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Matmul)
;
                        }
                        GeneratedField::TensorMultiply => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("tensorMultiply"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::TensorMultiply)
;
                        }
                        GeneratedField::Inverse => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("inverse"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Inverse)
;
                        }
                        GeneratedField::Determinant => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("determinant"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Determinant)
;
                        }
                        GeneratedField::Cholesky => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("cholesky"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Cholesky)
;
                        }
                        GeneratedField::CholeskyInverse => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("choleskyInverse"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::CholeskyInverse)
;
                        }
                        GeneratedField::Cast => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("cast"));
                            }
                            op__ = map_.next_value::<::std::option::Option<DType>>()?.map(|x| node::Op::Cast(x as i32));
                        }
                        GeneratedField::Where => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("where"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Where)
;
                        }
                        GeneratedField::Multiplexer => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("multiplexer"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Multiplexer)
;
                        }
                        GeneratedField::Cpt => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("cpt"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Cpt)
;
                        }
                        GeneratedField::SetWithMask => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("setWithMask"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::SetWithMask)
;
                        }
                        GeneratedField::ReplaceNan => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("replaceNan"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::ReplaceNan)
;
                        }
                        GeneratedField::Proxy => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("proxy"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Proxy)
;
                        }
                        GeneratedField::Assert => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("assert"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Assert)
;
                        }
                        GeneratedField::Distribution => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("distribution"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Distribution)
;
                        }
                        GeneratedField::Gather => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("gather"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Gather)
;
                        }
                        GeneratedField::ScatterAdd => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("scatterAdd"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::ScatterAdd)
;
                        }
                        GeneratedField::Stack => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stack"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Stack)
;
                        }
                        GeneratedField::Softmax => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("softmax"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Softmax)
;
                        }
                        GeneratedField::LogSoftmax => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("logSoftmax"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::LogSoftmax)
;
                        }
                    }
                }
                Ok(Node {
                    parents: parents__.unwrap_or_default(),
                    label: label__.unwrap_or_default(),
                    op: op__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.Node", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Permute {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.axes.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Permute", len)?;
        if !self.axes.is_empty() {
            struct_ser.serialize_field("axes", &self.axes.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Permute {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "axes",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Axes,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "axes" => Ok(GeneratedField::Axes),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Permute;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Permute")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Permute, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut axes__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Axes => {
                            if axes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axes"));
                            }
                            axes__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(Permute {
                    axes: axes__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Permute", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Predicate {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "PREDICATE_UNSPECIFIED",
            Self::IsNan => "PREDICATE_IS_NAN",
            Self::NotNan => "PREDICATE_NOT_NAN",
            Self::IsFinite => "PREDICATE_IS_FINITE",
            Self::IsInfinite => "PREDICATE_IS_INFINITE",
            Self::IsPositiveInfinity => "PREDICATE_IS_POSITIVE_INFINITY",
            Self::IsNegativeInfinity => "PREDICATE_IS_NEGATIVE_INFINITY",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for Predicate {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "PREDICATE_UNSPECIFIED",
            "PREDICATE_IS_NAN",
            "PREDICATE_NOT_NAN",
            "PREDICATE_IS_FINITE",
            "PREDICATE_IS_INFINITE",
            "PREDICATE_IS_POSITIVE_INFINITY",
            "PREDICATE_IS_NEGATIVE_INFINITY",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = Predicate;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "PREDICATE_UNSPECIFIED" => Ok(Predicate::Unspecified),
                    "PREDICATE_IS_NAN" => Ok(Predicate::IsNan),
                    "PREDICATE_NOT_NAN" => Ok(Predicate::NotNan),
                    "PREDICATE_IS_FINITE" => Ok(Predicate::IsFinite),
                    "PREDICATE_IS_INFINITE" => Ok(Predicate::IsInfinite),
                    "PREDICATE_IS_POSITIVE_INFINITY" => Ok(Predicate::IsPositiveInfinity),
                    "PREDICATE_IS_NEGATIVE_INFINITY" => Ok(Predicate::IsNegativeInfinity),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for Proxy {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.dtype != 0 {
            len += 1;
        }
        if !self.shape.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Proxy", len)?;
        if self.dtype != 0 {
            let v = DType::try_from(self.dtype)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.dtype)))?;
            struct_ser.serialize_field("dtype", &v)?;
        }
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Proxy {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "dtype",
            "shape",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Dtype,
            Shape,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "dtype" => Ok(GeneratedField::Dtype),
                            "shape" => Ok(GeneratedField::Shape),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Proxy;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Proxy")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Proxy, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut dtype__ = None;
                let mut shape__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Dtype => {
                            if dtype__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dtype"));
                            }
                            dtype__ = Some(map_.next_value::<DType>()? as i32);
                        }
                        GeneratedField::Shape => {
                            if shape__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(Proxy {
                    dtype: dtype__.unwrap_or_default(),
                    shape: shape__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Proxy", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Reduce {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind != 0 {
            len += 1;
        }
        if self.axes.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Reduce", len)?;
        if self.kind != 0 {
            let v = ReduceOp::try_from(self.kind)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.kind)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.axes.as_ref() {
            struct_ser.serialize_field("axes", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Reduce {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "axes",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Axes,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "kind" => Ok(GeneratedField::Kind),
                            "axes" => Ok(GeneratedField::Axes),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Reduce;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Reduce")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Reduce, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut axes__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = Some(map_.next_value::<ReduceOp>()? as i32);
                        }
                        GeneratedField::Axes => {
                            if axes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axes"));
                            }
                            axes__ = map_.next_value()?;
                        }
                    }
                }
                Ok(Reduce {
                    kind: kind__.unwrap_or_default(),
                    axes: axes__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.Reduce", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ReduceOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "REDUCE_OP_UNSPECIFIED",
            Self::Sum => "REDUCE_OP_SUM",
            Self::Product => "REDUCE_OP_PRODUCT",
            Self::Mean => "REDUCE_OP_MEAN",
            Self::Max => "REDUCE_OP_MAX",
            Self::Min => "REDUCE_OP_MIN",
            Self::StdDev => "REDUCE_OP_STD_DEV",
            Self::LogSumExp => "REDUCE_OP_LOG_SUM_EXP",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for ReduceOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "REDUCE_OP_UNSPECIFIED",
            "REDUCE_OP_SUM",
            "REDUCE_OP_PRODUCT",
            "REDUCE_OP_MEAN",
            "REDUCE_OP_MAX",
            "REDUCE_OP_MIN",
            "REDUCE_OP_STD_DEV",
            "REDUCE_OP_LOG_SUM_EXP",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = ReduceOp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "REDUCE_OP_UNSPECIFIED" => Ok(ReduceOp::Unspecified),
                    "REDUCE_OP_SUM" => Ok(ReduceOp::Sum),
                    "REDUCE_OP_PRODUCT" => Ok(ReduceOp::Product),
                    "REDUCE_OP_MEAN" => Ok(ReduceOp::Mean),
                    "REDUCE_OP_MAX" => Ok(ReduceOp::Max),
                    "REDUCE_OP_MIN" => Ok(ReduceOp::Min),
                    "REDUCE_OP_STD_DEV" => Ok(ReduceOp::StdDev),
                    "REDUCE_OP_LOG_SUM_EXP" => Ok(ReduceOp::LogSumExp),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for ReplaceNan {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.value != 0. {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.ReplaceNan", len)?;
        if self.value != 0. {
            struct_ser.serialize_field("value", &self.value)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ReplaceNan {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "value",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Value,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "value" => Ok(GeneratedField::Value),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ReplaceNan;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.ReplaceNan")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ReplaceNan, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut value__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Value => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("value"));
                            }
                            value__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(ReplaceNan {
                    value: value__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.ReplaceNan", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Reshape {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.shape.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Reshape", len)?;
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Reshape {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shape",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Shape,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "shape" => Ok(GeneratedField::Shape),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Reshape;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Reshape")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Reshape, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut shape__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Shape => {
                            if shape__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(Reshape {
                    shape: shape__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Reshape", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ScatterAdd {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.axis != 0 {
            len += 1;
        }
        if self.size != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.ScatterAdd", len)?;
        if self.axis != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&self.axis).as_str())?;
        }
        if self.size != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("size", ToString::to_string(&self.size).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ScatterAdd {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "axis",
            "size",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Axis,
            Size,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "axis" => Ok(GeneratedField::Axis),
                            "size" => Ok(GeneratedField::Size),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ScatterAdd;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.ScatterAdd")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ScatterAdd, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut axis__ = None;
                let mut size__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Size => {
                            if size__.is_some() {
                                return Err(serde::de::Error::duplicate_field("size"));
                            }
                            size__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(ScatterAdd {
                    axis: axis__.unwrap_or_default(),
                    size: size__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.ScatterAdd", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Slice {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.spec.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Slice", len)?;
        if !self.spec.is_empty() {
            struct_ser.serialize_field("spec", &self.spec)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Slice {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "spec",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Spec,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "spec" => Ok(GeneratedField::Spec),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Slice;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Slice")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Slice, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut spec__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Spec => {
                            if spec__.is_some() {
                                return Err(serde::de::Error::duplicate_field("spec"));
                            }
                            spec__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(Slice {
                    spec: spec__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Slice", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Softmax {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.axis != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Softmax", len)?;
        if self.axis != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&self.axis).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Softmax {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "axis",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Axis,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "axis" => Ok(GeneratedField::Axis),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Softmax;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Softmax")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Softmax, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut axis__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Softmax {
                    axis: axis__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Softmax", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Stack {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.axis != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Stack", len)?;
        if self.axis != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("axis", ToString::to_string(&self.axis).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Stack {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "axis",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Axis,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "axis" => Ok(GeneratedField::Axis),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Stack;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Stack")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Stack, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut axis__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Axis => {
                            if axis__.is_some() {
                                return Err(serde::de::Error::duplicate_field("axis"));
                            }
                            axis__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Stack {
                    axis: axis__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Stack", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for StoredValue {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.node != 0 {
            len += 1;
        }
        if self.value.is_some() {
            len += 1;
        }
        if self.observed {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.StoredValue", len)?;
        if self.node != 0 {
            struct_ser.serialize_field("node", &self.node)?;
        }
        if let Some(v) = self.value.as_ref() {
            struct_ser.serialize_field("value", v)?;
        }
        if self.observed {
            struct_ser.serialize_field("observed", &self.observed)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for StoredValue {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "node",
            "value",
            "observed",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Node,
            Value,
            Observed,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "node" => Ok(GeneratedField::Node),
                            "value" => Ok(GeneratedField::Value),
                            "observed" => Ok(GeneratedField::Observed),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = StoredValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.StoredValue")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<StoredValue, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut node__ = None;
                let mut value__ = None;
                let mut observed__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Node => {
                            if node__.is_some() {
                                return Err(serde::de::Error::duplicate_field("node"));
                            }
                            node__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Value => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("value"));
                            }
                            value__ = map_.next_value()?;
                        }
                        GeneratedField::Observed => {
                            if observed__.is_some() {
                                return Err(serde::de::Error::duplicate_field("observed"));
                            }
                            observed__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(StoredValue {
                    node: node__.unwrap_or_default(),
                    value: value__,
                    observed: observed__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.StoredValue", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Take {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.index.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Take", len)?;
        if !self.index.is_empty() {
            struct_ser.serialize_field("index", &self.index.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Take {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "index",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Index,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "index" => Ok(GeneratedField::Index),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Take;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Take")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Take, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut index__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Index => {
                            if index__.is_some() {
                                return Err(serde::de::Error::duplicate_field("index"));
                            }
                            index__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(Take {
                    index: index__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Take", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Tensor {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.shape.is_empty() {
            len += 1;
        }
        if self.values.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Tensor", len)?;
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if let Some(v) = self.values.as_ref() {
            match v {
                tensor::Values::F64(v) => {
                    struct_ser.serialize_field("f64", v)?;
                }
                tensor::Values::I64(v) => {
                    struct_ser.serialize_field("i64", v)?;
                }
                tensor::Values::Bool(v) => {
                    struct_ser.serialize_field("bool", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Tensor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shape",
            "f64",
            "i64",
            "bool",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Shape,
            F64,
            I64,
            Bool,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "shape" => Ok(GeneratedField::Shape),
                            "f64" => Ok(GeneratedField::F64),
                            "i64" => Ok(GeneratedField::I64),
                            "bool" => Ok(GeneratedField::Bool),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Tensor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Tensor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Tensor, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut shape__ = None;
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Shape => {
                            if shape__.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::F64 => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("f64"));
                            }
                            values__ = map_.next_value::<::std::option::Option<_>>()?.map(tensor::Values::F64)
;
                        }
                        GeneratedField::I64 => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("i64"));
                            }
                            values__ = map_.next_value::<::std::option::Option<_>>()?.map(tensor::Values::I64)
;
                        }
                        GeneratedField::Bool => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bool"));
                            }
                            values__ = map_.next_value::<::std::option::Option<_>>()?.map(tensor::Values::Bool)
;
                        }
                    }
                }
                Ok(Tensor {
                    shape: shape__.unwrap_or_default(),
                    values: values__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.Tensor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TensorMultiply {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.left_axes.is_empty() {
            len += 1;
        }
        if !self.right_axes.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.TensorMultiply", len)?;
        if !self.left_axes.is_empty() {
            struct_ser.serialize_field("leftAxes", &self.left_axes.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.right_axes.is_empty() {
            struct_ser.serialize_field("rightAxes", &self.right_axes.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TensorMultiply {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "left_axes",
            "leftAxes",
            "right_axes",
            "rightAxes",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            LeftAxes,
            RightAxes,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "leftAxes" | "left_axes" => Ok(GeneratedField::LeftAxes),
                            "rightAxes" | "right_axes" => Ok(GeneratedField::RightAxes),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TensorMultiply;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.TensorMultiply")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TensorMultiply, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut left_axes__ = None;
                let mut right_axes__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::LeftAxes => {
                            if left_axes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("leftAxes"));
                            }
                            left_axes__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::RightAxes => {
                            if right_axes__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rightAxes"));
                            }
                            right_axes__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(TensorMultiply {
                    left_axes: left_axes__.unwrap_or_default(),
                    right_axes: right_axes__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.TensorMultiply", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Triangle {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.k != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Triangle", len)?;
        if self.k != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("k", ToString::to_string(&self.k).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Triangle {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "k",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            K,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "k" => Ok(GeneratedField::K),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Triangle;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Triangle")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Triangle, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut k__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::K => {
                            if k__.is_some() {
                                return Err(serde::de::Error::duplicate_field("k"));
                            }
                            k__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Triangle {
                    k: k__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Triangle", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for TrianglePart {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.upper {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.TrianglePart", len)?;
        if self.upper {
            struct_ser.serialize_field("upper", &self.upper)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for TrianglePart {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "upper",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Upper,
        }
        impl<'de> serde::Deserialize<'de> for GeneratedField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<GeneratedField, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct GeneratedVisitor;

                impl serde::de::Visitor<'_> for GeneratedVisitor {
                    type Value = GeneratedField;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(formatter, "expected one of: {:?}", &FIELDS)
                    }

                    #[allow(unused_variables)]
                    fn visit_str<E>(self, value: &str) -> std::result::Result<GeneratedField, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "upper" => Ok(GeneratedField::Upper),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = TrianglePart;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.TrianglePart")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<TrianglePart, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut upper__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Upper => {
                            if upper__.is_some() {
                                return Err(serde::de::Error::duplicate_field("upper"));
                            }
                            upper__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(TrianglePart {
                    upper: upper__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.TrianglePart", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for UnaryOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "UNARY_OP_UNSPECIFIED",
            Self::Abs => "UNARY_OP_ABS",
            Self::Neg => "UNARY_OP_NEG",
            Self::Sign => "UNARY_OP_SIGN",
            Self::Reciprocal => "UNARY_OP_RECIPROCAL",
            Self::Exp => "UNARY_OP_EXP",
            Self::Exp2 => "UNARY_OP_EXP2",
            Self::ExpM1 => "UNARY_OP_EXP_M1",
            Self::Ln => "UNARY_OP_LN",
            Self::Ln1p => "UNARY_OP_LN1P",
            Self::Log2 => "UNARY_OP_LOG2",
            Self::Log10 => "UNARY_OP_LOG10",
            Self::Sqrt => "UNARY_OP_SQRT",
            Self::Sin => "UNARY_OP_SIN",
            Self::Cos => "UNARY_OP_COS",
            Self::Tan => "UNARY_OP_TAN",
            Self::Asin => "UNARY_OP_ASIN",
            Self::Acos => "UNARY_OP_ACOS",
            Self::Atan => "UNARY_OP_ATAN",
            Self::Sinh => "UNARY_OP_SINH",
            Self::Cosh => "UNARY_OP_COSH",
            Self::Tanh => "UNARY_OP_TANH",
            Self::Asinh => "UNARY_OP_ASINH",
            Self::Acosh => "UNARY_OP_ACOSH",
            Self::Atanh => "UNARY_OP_ATANH",
            Self::Sigmoid => "UNARY_OP_SIGMOID",
            Self::Floor => "UNARY_OP_FLOOR",
            Self::Ceil => "UNARY_OP_CEIL",
            Self::Round => "UNARY_OP_ROUND",
            Self::LnGamma => "UNARY_OP_LN_GAMMA",
            Self::Digamma => "UNARY_OP_DIGAMMA",
            Self::Trigamma => "UNARY_OP_TRIGAMMA",
            Self::Erf => "UNARY_OP_ERF",
            Self::Erfc => "UNARY_OP_ERFC",
            Self::ErfInv => "UNARY_OP_ERF_INV",
            Self::LogNdtr => "UNARY_OP_LOG_NDTR",
            Self::Ndtri => "UNARY_OP_NDTRI",
            Self::Softplus => "UNARY_OP_SOFTPLUS",
            Self::Logit => "UNARY_OP_LOGIT",
            Self::Log1mExp => "UNARY_OP_LOG1M_EXP",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for UnaryOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "UNARY_OP_UNSPECIFIED",
            "UNARY_OP_ABS",
            "UNARY_OP_NEG",
            "UNARY_OP_SIGN",
            "UNARY_OP_RECIPROCAL",
            "UNARY_OP_EXP",
            "UNARY_OP_EXP2",
            "UNARY_OP_EXP_M1",
            "UNARY_OP_LN",
            "UNARY_OP_LN1P",
            "UNARY_OP_LOG2",
            "UNARY_OP_LOG10",
            "UNARY_OP_SQRT",
            "UNARY_OP_SIN",
            "UNARY_OP_COS",
            "UNARY_OP_TAN",
            "UNARY_OP_ASIN",
            "UNARY_OP_ACOS",
            "UNARY_OP_ATAN",
            "UNARY_OP_SINH",
            "UNARY_OP_COSH",
            "UNARY_OP_TANH",
            "UNARY_OP_ASINH",
            "UNARY_OP_ACOSH",
            "UNARY_OP_ATANH",
            "UNARY_OP_SIGMOID",
            "UNARY_OP_FLOOR",
            "UNARY_OP_CEIL",
            "UNARY_OP_ROUND",
            "UNARY_OP_LN_GAMMA",
            "UNARY_OP_DIGAMMA",
            "UNARY_OP_TRIGAMMA",
            "UNARY_OP_ERF",
            "UNARY_OP_ERFC",
            "UNARY_OP_ERF_INV",
            "UNARY_OP_LOG_NDTR",
            "UNARY_OP_NDTRI",
            "UNARY_OP_SOFTPLUS",
            "UNARY_OP_LOGIT",
            "UNARY_OP_LOG1M_EXP",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = UnaryOp;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "expected one of: {:?}", &FIELDS)
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Signed(v), &self)
                    })
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                i32::try_from(v)
                    .ok()
                    .and_then(|x| x.try_into().ok())
                    .ok_or_else(|| {
                        serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(v), &self)
                    })
            }

            fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value {
                    "UNARY_OP_UNSPECIFIED" => Ok(UnaryOp::Unspecified),
                    "UNARY_OP_ABS" => Ok(UnaryOp::Abs),
                    "UNARY_OP_NEG" => Ok(UnaryOp::Neg),
                    "UNARY_OP_SIGN" => Ok(UnaryOp::Sign),
                    "UNARY_OP_RECIPROCAL" => Ok(UnaryOp::Reciprocal),
                    "UNARY_OP_EXP" => Ok(UnaryOp::Exp),
                    "UNARY_OP_EXP2" => Ok(UnaryOp::Exp2),
                    "UNARY_OP_EXP_M1" => Ok(UnaryOp::ExpM1),
                    "UNARY_OP_LN" => Ok(UnaryOp::Ln),
                    "UNARY_OP_LN1P" => Ok(UnaryOp::Ln1p),
                    "UNARY_OP_LOG2" => Ok(UnaryOp::Log2),
                    "UNARY_OP_LOG10" => Ok(UnaryOp::Log10),
                    "UNARY_OP_SQRT" => Ok(UnaryOp::Sqrt),
                    "UNARY_OP_SIN" => Ok(UnaryOp::Sin),
                    "UNARY_OP_COS" => Ok(UnaryOp::Cos),
                    "UNARY_OP_TAN" => Ok(UnaryOp::Tan),
                    "UNARY_OP_ASIN" => Ok(UnaryOp::Asin),
                    "UNARY_OP_ACOS" => Ok(UnaryOp::Acos),
                    "UNARY_OP_ATAN" => Ok(UnaryOp::Atan),
                    "UNARY_OP_SINH" => Ok(UnaryOp::Sinh),
                    "UNARY_OP_COSH" => Ok(UnaryOp::Cosh),
                    "UNARY_OP_TANH" => Ok(UnaryOp::Tanh),
                    "UNARY_OP_ASINH" => Ok(UnaryOp::Asinh),
                    "UNARY_OP_ACOSH" => Ok(UnaryOp::Acosh),
                    "UNARY_OP_ATANH" => Ok(UnaryOp::Atanh),
                    "UNARY_OP_SIGMOID" => Ok(UnaryOp::Sigmoid),
                    "UNARY_OP_FLOOR" => Ok(UnaryOp::Floor),
                    "UNARY_OP_CEIL" => Ok(UnaryOp::Ceil),
                    "UNARY_OP_ROUND" => Ok(UnaryOp::Round),
                    "UNARY_OP_LN_GAMMA" => Ok(UnaryOp::LnGamma),
                    "UNARY_OP_DIGAMMA" => Ok(UnaryOp::Digamma),
                    "UNARY_OP_TRIGAMMA" => Ok(UnaryOp::Trigamma),
                    "UNARY_OP_ERF" => Ok(UnaryOp::Erf),
                    "UNARY_OP_ERFC" => Ok(UnaryOp::Erfc),
                    "UNARY_OP_ERF_INV" => Ok(UnaryOp::ErfInv),
                    "UNARY_OP_LOG_NDTR" => Ok(UnaryOp::LogNdtr),
                    "UNARY_OP_NDTRI" => Ok(UnaryOp::Ndtri),
                    "UNARY_OP_SOFTPLUS" => Ok(UnaryOp::Softplus),
                    "UNARY_OP_LOGIT" => Ok(UnaryOp::Logit),
                    "UNARY_OP_LOG1M_EXP" => Ok(UnaryOp::Log1mExp),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
