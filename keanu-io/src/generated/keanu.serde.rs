impl serde::Serialize for Adjacency {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.areas != 0 {
            len += 1;
        }
        if !self.edges.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Adjacency", len)?;
        if self.areas != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("areas", ToString::to_string(&self.areas).as_str())?;
        }
        if !self.edges.is_empty() {
            struct_ser.serialize_field("edges", &self.edges.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Adjacency {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "areas",
            "edges",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Areas,
            Edges,
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
                            "areas" => Ok(GeneratedField::Areas),
                            "edges" => Ok(GeneratedField::Edges),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Adjacency;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Adjacency")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Adjacency, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut areas__ = None;
                let mut edges__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Areas => {
                            if areas__.is_some() {
                                return Err(serde::de::Error::duplicate_field("areas"));
                            }
                            areas__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Edges => {
                            if edges__.is_some() {
                                return Err(serde::de::Error::duplicate_field("edges"));
                            }
                            edges__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                    }
                }
                Ok(Adjacency {
                    areas: areas__.unwrap_or_default(),
                    edges: edges__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Adjacency", FIELDS, GeneratedVisitor)
    }
}
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
impl serde::Serialize for Bessel {
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
        if self.order != 0. {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Bessel", len)?;
        if self.kind != 0 {
            let v = BesselOp::try_from(self.kind)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.kind)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if self.order != 0. {
            struct_ser.serialize_field("order", &self.order)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Bessel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "order",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Order,
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
                            "order" => Ok(GeneratedField::Order),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Bessel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Bessel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Bessel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut order__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = Some(map_.next_value::<BesselOp>()? as i32);
                        }
                        GeneratedField::Order => {
                            if order__.is_some() {
                                return Err(serde::de::Error::duplicate_field("order"));
                            }
                            order__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Bessel {
                    kind: kind__.unwrap_or_default(),
                    order: order__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Bessel", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for BesselOp {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "BESSEL_OP_UNSPECIFIED",
            Self::K => "BESSEL_OP_K",
            Self::LogK => "BESSEL_OP_LOG_K",
            Self::LogI => "BESSEL_OP_LOG_I",
            Self::Matern => "BESSEL_OP_MATERN",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for BesselOp {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "BESSEL_OP_UNSPECIFIED",
            "BESSEL_OP_K",
            "BESSEL_OP_LOG_K",
            "BESSEL_OP_LOG_I",
            "BESSEL_OP_MATERN",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = BesselOp;

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
                    "BESSEL_OP_UNSPECIFIED" => Ok(BesselOp::Unspecified),
                    "BESSEL_OP_K" => Ok(BesselOp::K),
                    "BESSEL_OP_LOG_K" => Ok(BesselOp::LogK),
                    "BESSEL_OP_LOG_I" => Ok(BesselOp::LogI),
                    "BESSEL_OP_MATERN" => Ok(BesselOp::Matern),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
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
            Self::OwensT => "BINARY_OP_OWENS_T",
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
            "BINARY_OP_OWENS_T",
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
                    "BINARY_OP_OWENS_T" => Ok(BinaryOp::OwensT),
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
        if self.base.is_some() {
            len += 1;
        }
        if self.adjacency.is_some() {
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
        if let Some(v) = self.base.as_ref() {
            struct_ser.serialize_field("base", v)?;
        }
        if let Some(v) = self.adjacency.as_ref() {
            struct_ser.serialize_field("adjacency", v)?;
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
            "base",
            "adjacency",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Shape,
            Kde,
            Base,
            Adjacency,
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
                            "base" => Ok(GeneratedField::Base),
                            "adjacency" => Ok(GeneratedField::Adjacency),
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
                let mut base__ = None;
                let mut adjacency__ = None;
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
                        GeneratedField::Base => {
                            if base__.is_some() {
                                return Err(serde::de::Error::duplicate_field("base"));
                            }
                            base__ = map_.next_value()?;
                        }
                        GeneratedField::Adjacency => {
                            if adjacency__.is_some() {
                                return Err(serde::de::Error::duplicate_field("adjacency"));
                            }
                            adjacency__ = map_.next_value()?;
                        }
                    }
                }
                Ok(Distribution {
                    kind: kind__.unwrap_or_default(),
                    shape: shape__.unwrap_or_default(),
                    kde: kde__,
                    base: base__,
                    adjacency: adjacency__,
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
            Self::Weibull => "DISTRIBUTION_KIND_WEIBULL",
            Self::VonMises => "DISTRIBUTION_KIND_VON_MISES",
            Self::NegativeBinomial => "DISTRIBUTION_KIND_NEGATIVE_BINOMIAL",
            Self::OrderedLogistic => "DISTRIBUTION_KIND_ORDERED_LOGISTIC",
            Self::Truncated => "DISTRIBUTION_KIND_TRUNCATED",
            Self::Censored => "DISTRIBUTION_KIND_CENSORED",
            Self::Mixture => "DISTRIBUTION_KIND_MIXTURE",
            Self::HiddenMarkovModel => "DISTRIBUTION_KIND_HIDDEN_MARKOV_MODEL",
            Self::LinearGaussianStateSpace => "DISTRIBUTION_KIND_LINEAR_GAUSSIAN_STATE_SPACE",
            Self::ZeroSumNormal => "DISTRIBUTION_KIND_ZERO_SUM_NORMAL",
            Self::BetaBinomial => "DISTRIBUTION_KIND_BETA_BINOMIAL",
            Self::BernoulliLogit => "DISTRIBUTION_KIND_BERNOULLI_LOGIT",
            Self::BinomialLogit => "DISTRIBUTION_KIND_BINOMIAL_LOGIT",
            Self::PoissonLog => "DISTRIBUTION_KIND_POISSON_LOG",
            Self::CategoricalLogit => "DISTRIBUTION_KIND_CATEGORICAL_LOGIT",
            Self::ZeroInflated => "DISTRIBUTION_KIND_ZERO_INFLATED",
            Self::Hurdle => "DISTRIBUTION_KIND_HURDLE",
            Self::GaussianRandomWalk => "DISTRIBUTION_KIND_GAUSSIAN_RANDOM_WALK",
            Self::Autoregressive => "DISTRIBUTION_KIND_AUTOREGRESSIVE",
            Self::Garch11 => "DISTRIBUTION_KIND_GARCH11",
            Self::Icar => "DISTRIBUTION_KIND_ICAR",
            Self::Car => "DISTRIBUTION_KIND_CAR",
            Self::SkewNormal => "DISTRIBUTION_KIND_SKEW_NORMAL",
            Self::ExGaussian => "DISTRIBUTION_KIND_EX_GAUSSIAN",
            Self::Gumbel => "DISTRIBUTION_KIND_GUMBEL",
            Self::GeneralizedExtremeValue => "DISTRIBUTION_KIND_GENERALIZED_EXTREME_VALUE",
            Self::GeneralizedPareto => "DISTRIBUTION_KIND_GENERALIZED_PARETO",
            Self::InverseGaussian => "DISTRIBUTION_KIND_INVERSE_GAUSSIAN",
            Self::Kumaraswamy => "DISTRIBUTION_KIND_KUMARASWAMY",
            Self::LogitNormal => "DISTRIBUTION_KIND_LOGIT_NORMAL",
            Self::HalfStudentT => "DISTRIBUTION_KIND_HALF_STUDENT_T",
            Self::AsymmetricLaplace => "DISTRIBUTION_KIND_ASYMMETRIC_LAPLACE",
            Self::LogLogistic => "DISTRIBUTION_KIND_LOG_LOGISTIC",
            Self::Rice => "DISTRIBUTION_KIND_RICE",
            Self::GeneralizedGamma => "DISTRIBUTION_KIND_GENERALIZED_GAMMA",
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
            "DISTRIBUTION_KIND_WEIBULL",
            "DISTRIBUTION_KIND_VON_MISES",
            "DISTRIBUTION_KIND_NEGATIVE_BINOMIAL",
            "DISTRIBUTION_KIND_ORDERED_LOGISTIC",
            "DISTRIBUTION_KIND_TRUNCATED",
            "DISTRIBUTION_KIND_CENSORED",
            "DISTRIBUTION_KIND_MIXTURE",
            "DISTRIBUTION_KIND_HIDDEN_MARKOV_MODEL",
            "DISTRIBUTION_KIND_LINEAR_GAUSSIAN_STATE_SPACE",
            "DISTRIBUTION_KIND_ZERO_SUM_NORMAL",
            "DISTRIBUTION_KIND_BETA_BINOMIAL",
            "DISTRIBUTION_KIND_BERNOULLI_LOGIT",
            "DISTRIBUTION_KIND_BINOMIAL_LOGIT",
            "DISTRIBUTION_KIND_POISSON_LOG",
            "DISTRIBUTION_KIND_CATEGORICAL_LOGIT",
            "DISTRIBUTION_KIND_ZERO_INFLATED",
            "DISTRIBUTION_KIND_HURDLE",
            "DISTRIBUTION_KIND_GAUSSIAN_RANDOM_WALK",
            "DISTRIBUTION_KIND_AUTOREGRESSIVE",
            "DISTRIBUTION_KIND_GARCH11",
            "DISTRIBUTION_KIND_ICAR",
            "DISTRIBUTION_KIND_CAR",
            "DISTRIBUTION_KIND_SKEW_NORMAL",
            "DISTRIBUTION_KIND_EX_GAUSSIAN",
            "DISTRIBUTION_KIND_GUMBEL",
            "DISTRIBUTION_KIND_GENERALIZED_EXTREME_VALUE",
            "DISTRIBUTION_KIND_GENERALIZED_PARETO",
            "DISTRIBUTION_KIND_INVERSE_GAUSSIAN",
            "DISTRIBUTION_KIND_KUMARASWAMY",
            "DISTRIBUTION_KIND_LOGIT_NORMAL",
            "DISTRIBUTION_KIND_HALF_STUDENT_T",
            "DISTRIBUTION_KIND_ASYMMETRIC_LAPLACE",
            "DISTRIBUTION_KIND_LOG_LOGISTIC",
            "DISTRIBUTION_KIND_RICE",
            "DISTRIBUTION_KIND_GENERALIZED_GAMMA",
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
                    "DISTRIBUTION_KIND_WEIBULL" => Ok(DistributionKind::Weibull),
                    "DISTRIBUTION_KIND_VON_MISES" => Ok(DistributionKind::VonMises),
                    "DISTRIBUTION_KIND_NEGATIVE_BINOMIAL" => Ok(DistributionKind::NegativeBinomial),
                    "DISTRIBUTION_KIND_ORDERED_LOGISTIC" => Ok(DistributionKind::OrderedLogistic),
                    "DISTRIBUTION_KIND_TRUNCATED" => Ok(DistributionKind::Truncated),
                    "DISTRIBUTION_KIND_CENSORED" => Ok(DistributionKind::Censored),
                    "DISTRIBUTION_KIND_MIXTURE" => Ok(DistributionKind::Mixture),
                    "DISTRIBUTION_KIND_HIDDEN_MARKOV_MODEL" => Ok(DistributionKind::HiddenMarkovModel),
                    "DISTRIBUTION_KIND_LINEAR_GAUSSIAN_STATE_SPACE" => Ok(DistributionKind::LinearGaussianStateSpace),
                    "DISTRIBUTION_KIND_ZERO_SUM_NORMAL" => Ok(DistributionKind::ZeroSumNormal),
                    "DISTRIBUTION_KIND_BETA_BINOMIAL" => Ok(DistributionKind::BetaBinomial),
                    "DISTRIBUTION_KIND_BERNOULLI_LOGIT" => Ok(DistributionKind::BernoulliLogit),
                    "DISTRIBUTION_KIND_BINOMIAL_LOGIT" => Ok(DistributionKind::BinomialLogit),
                    "DISTRIBUTION_KIND_POISSON_LOG" => Ok(DistributionKind::PoissonLog),
                    "DISTRIBUTION_KIND_CATEGORICAL_LOGIT" => Ok(DistributionKind::CategoricalLogit),
                    "DISTRIBUTION_KIND_ZERO_INFLATED" => Ok(DistributionKind::ZeroInflated),
                    "DISTRIBUTION_KIND_HURDLE" => Ok(DistributionKind::Hurdle),
                    "DISTRIBUTION_KIND_GAUSSIAN_RANDOM_WALK" => Ok(DistributionKind::GaussianRandomWalk),
                    "DISTRIBUTION_KIND_AUTOREGRESSIVE" => Ok(DistributionKind::Autoregressive),
                    "DISTRIBUTION_KIND_GARCH11" => Ok(DistributionKind::Garch11),
                    "DISTRIBUTION_KIND_ICAR" => Ok(DistributionKind::Icar),
                    "DISTRIBUTION_KIND_CAR" => Ok(DistributionKind::Car),
                    "DISTRIBUTION_KIND_SKEW_NORMAL" => Ok(DistributionKind::SkewNormal),
                    "DISTRIBUTION_KIND_EX_GAUSSIAN" => Ok(DistributionKind::ExGaussian),
                    "DISTRIBUTION_KIND_GUMBEL" => Ok(DistributionKind::Gumbel),
                    "DISTRIBUTION_KIND_GENERALIZED_EXTREME_VALUE" => Ok(DistributionKind::GeneralizedExtremeValue),
                    "DISTRIBUTION_KIND_GENERALIZED_PARETO" => Ok(DistributionKind::GeneralizedPareto),
                    "DISTRIBUTION_KIND_INVERSE_GAUSSIAN" => Ok(DistributionKind::InverseGaussian),
                    "DISTRIBUTION_KIND_KUMARASWAMY" => Ok(DistributionKind::Kumaraswamy),
                    "DISTRIBUTION_KIND_LOGIT_NORMAL" => Ok(DistributionKind::LogitNormal),
                    "DISTRIBUTION_KIND_HALF_STUDENT_T" => Ok(DistributionKind::HalfStudentT),
                    "DISTRIBUTION_KIND_ASYMMETRIC_LAPLACE" => Ok(DistributionKind::AsymmetricLaplace),
                    "DISTRIBUTION_KIND_LOG_LOGISTIC" => Ok(DistributionKind::LogLogistic),
                    "DISTRIBUTION_KIND_RICE" => Ok(DistributionKind::Rice),
                    "DISTRIBUTION_KIND_GENERALIZED_GAMMA" => Ok(DistributionKind::GeneralizedGamma),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for DistributionType {
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
        if self.kde.is_some() {
            len += 1;
        }
        if self.base.is_some() {
            len += 1;
        }
        if self.adjacency.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.DistributionType", len)?;
        if self.kind != 0 {
            let v = DistributionKind::try_from(self.kind)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.kind)))?;
            struct_ser.serialize_field("kind", &v)?;
        }
        if let Some(v) = self.kde.as_ref() {
            struct_ser.serialize_field("kde", v)?;
        }
        if let Some(v) = self.base.as_ref() {
            struct_ser.serialize_field("base", v)?;
        }
        if let Some(v) = self.adjacency.as_ref() {
            struct_ser.serialize_field("adjacency", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for DistributionType {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
            "kde",
            "base",
            "adjacency",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
            Kde,
            Base,
            Adjacency,
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
                            "kde" => Ok(GeneratedField::Kde),
                            "base" => Ok(GeneratedField::Base),
                            "adjacency" => Ok(GeneratedField::Adjacency),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = DistributionType;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.DistributionType")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<DistributionType, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                let mut kde__ = None;
                let mut base__ = None;
                let mut adjacency__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = Some(map_.next_value::<DistributionKind>()? as i32);
                        }
                        GeneratedField::Kde => {
                            if kde__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kde"));
                            }
                            kde__ = map_.next_value()?;
                        }
                        GeneratedField::Base => {
                            if base__.is_some() {
                                return Err(serde::de::Error::duplicate_field("base"));
                            }
                            base__ = map_.next_value()?;
                        }
                        GeneratedField::Adjacency => {
                            if adjacency__.is_some() {
                                return Err(serde::de::Error::duplicate_field("adjacency"));
                            }
                            adjacency__ = map_.next_value()?;
                        }
                    }
                }
                Ok(DistributionType {
                    kind: kind__.unwrap_or_default(),
                    kde: kde__,
                    base: base__,
                    adjacency: adjacency__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.DistributionType", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for LogDensity {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.kind.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.LogDensity", len)?;
        if let Some(v) = self.kind.as_ref() {
            struct_ser.serialize_field("kind", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for LogDensity {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "kind",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Kind,
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
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = LogDensity;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.LogDensity")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<LogDensity, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut kind__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Kind => {
                            if kind__.is_some() {
                                return Err(serde::de::Error::duplicate_field("kind"));
                            }
                            kind__ = map_.next_value()?;
                        }
                    }
                }
                Ok(LogDensity {
                    kind: kind__,
                })
            }
        }
        deserializer.deserialize_struct("keanu.LogDensity", FIELDS, GeneratedVisitor)
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
                node::Op::SolveTriangular(v) => {
                    struct_ser.serialize_field("solveTriangular", v)?;
                }
                node::Op::Solve(v) => {
                    struct_ser.serialize_field("solve", v)?;
                }
                node::Op::LogAbsDeterminant(v) => {
                    struct_ser.serialize_field("logAbsDeterminant", v)?;
                }
                node::Op::DeterminantSign(v) => {
                    struct_ser.serialize_field("determinantSign", v)?;
                }
                node::Op::Eigvalsh(v) => {
                    struct_ser.serialize_field("eigvalsh", v)?;
                }
                node::Op::EighVectors(v) => {
                    struct_ser.serialize_field("eighVectors", v)?;
                }
                node::Op::Factor(v) => {
                    struct_ser.serialize_field("factor", v)?;
                }
                node::Op::LogDensity(v) => {
                    struct_ser.serialize_field("logDensity", v)?;
                }
                node::Op::MatrixExp(v) => {
                    struct_ser.serialize_field("matrixExp", v)?;
                }
                node::Op::SparseMatmul(v) => {
                    struct_ser.serialize_field("sparseMatmul", v)?;
                }
                node::Op::Bessel(v) => {
                    struct_ser.serialize_field("bessel", v)?;
                }
                node::Op::Scan(v) => {
                    struct_ser.serialize_field("scan", v)?;
                }
                node::Op::Ode(v) => {
                    struct_ser.serialize_field("ode", v)?;
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
            "solve_triangular",
            "solveTriangular",
            "solve",
            "log_abs_determinant",
            "logAbsDeterminant",
            "determinant_sign",
            "determinantSign",
            "eigvalsh",
            "eigh_vectors",
            "eighVectors",
            "factor",
            "log_density",
            "logDensity",
            "matrix_exp",
            "matrixExp",
            "sparse_matmul",
            "sparseMatmul",
            "bessel",
            "scan",
            "ode",
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
            SolveTriangular,
            Solve,
            LogAbsDeterminant,
            DeterminantSign,
            Eigvalsh,
            EighVectors,
            Factor,
            LogDensity,
            MatrixExp,
            SparseMatmul,
            Bessel,
            Scan,
            Ode,
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
                            "solveTriangular" | "solve_triangular" => Ok(GeneratedField::SolveTriangular),
                            "solve" => Ok(GeneratedField::Solve),
                            "logAbsDeterminant" | "log_abs_determinant" => Ok(GeneratedField::LogAbsDeterminant),
                            "determinantSign" | "determinant_sign" => Ok(GeneratedField::DeterminantSign),
                            "eigvalsh" => Ok(GeneratedField::Eigvalsh),
                            "eighVectors" | "eigh_vectors" => Ok(GeneratedField::EighVectors),
                            "factor" => Ok(GeneratedField::Factor),
                            "logDensity" | "log_density" => Ok(GeneratedField::LogDensity),
                            "matrixExp" | "matrix_exp" => Ok(GeneratedField::MatrixExp),
                            "sparseMatmul" | "sparse_matmul" => Ok(GeneratedField::SparseMatmul),
                            "bessel" => Ok(GeneratedField::Bessel),
                            "scan" => Ok(GeneratedField::Scan),
                            "ode" => Ok(GeneratedField::Ode),
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
                        GeneratedField::SolveTriangular => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("solveTriangular"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::SolveTriangular)
;
                        }
                        GeneratedField::Solve => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("solve"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Solve)
;
                        }
                        GeneratedField::LogAbsDeterminant => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("logAbsDeterminant"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::LogAbsDeterminant)
;
                        }
                        GeneratedField::DeterminantSign => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("determinantSign"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::DeterminantSign)
;
                        }
                        GeneratedField::Eigvalsh => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("eigvalsh"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Eigvalsh)
;
                        }
                        GeneratedField::EighVectors => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("eighVectors"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::EighVectors)
;
                        }
                        GeneratedField::Factor => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("factor"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Factor)
;
                        }
                        GeneratedField::LogDensity => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("logDensity"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::LogDensity)
;
                        }
                        GeneratedField::MatrixExp => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("matrixExp"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::MatrixExp)
;
                        }
                        GeneratedField::SparseMatmul => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("sparseMatmul"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::SparseMatmul)
;
                        }
                        GeneratedField::Bessel => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("bessel"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Bessel)
;
                        }
                        GeneratedField::Scan => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("scan"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Scan)
;
                        }
                        GeneratedField::Ode => {
                            if op__.is_some() {
                                return Err(serde::de::Error::duplicate_field("ode"));
                            }
                            op__ = map_.next_value::<::std::option::Option<_>>()?.map(node::Op::Ode)
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
impl serde::Serialize for Ode {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.body.is_some() {
            len += 1;
        }
        if self.time != 0 {
            len += 1;
        }
        if self.state != 0 {
            len += 1;
        }
        if !self.parameters.is_empty() {
            len += 1;
        }
        if self.output != 0 {
            len += 1;
        }
        if self.solver != 0 {
            len += 1;
        }
        if self.rtol != 0. {
            len += 1;
        }
        if self.atol != 0. {
            len += 1;
        }
        if self.max_steps != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Ode", len)?;
        if let Some(v) = self.body.as_ref() {
            struct_ser.serialize_field("body", v)?;
        }
        if self.time != 0 {
            struct_ser.serialize_field("time", &self.time)?;
        }
        if self.state != 0 {
            struct_ser.serialize_field("state", &self.state)?;
        }
        if !self.parameters.is_empty() {
            struct_ser.serialize_field("parameters", &self.parameters)?;
        }
        if self.output != 0 {
            struct_ser.serialize_field("output", &self.output)?;
        }
        if self.solver != 0 {
            let v = OdeSolver::try_from(self.solver)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.solver)))?;
            struct_ser.serialize_field("solver", &v)?;
        }
        if self.rtol != 0. {
            struct_ser.serialize_field("rtol", &self.rtol)?;
        }
        if self.atol != 0. {
            struct_ser.serialize_field("atol", &self.atol)?;
        }
        if self.max_steps != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("maxSteps", ToString::to_string(&self.max_steps).as_str())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Ode {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "body",
            "time",
            "state",
            "parameters",
            "output",
            "solver",
            "rtol",
            "atol",
            "max_steps",
            "maxSteps",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Body,
            Time,
            State,
            Parameters,
            Output,
            Solver,
            Rtol,
            Atol,
            MaxSteps,
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
                            "body" => Ok(GeneratedField::Body),
                            "time" => Ok(GeneratedField::Time),
                            "state" => Ok(GeneratedField::State),
                            "parameters" => Ok(GeneratedField::Parameters),
                            "output" => Ok(GeneratedField::Output),
                            "solver" => Ok(GeneratedField::Solver),
                            "rtol" => Ok(GeneratedField::Rtol),
                            "atol" => Ok(GeneratedField::Atol),
                            "maxSteps" | "max_steps" => Ok(GeneratedField::MaxSteps),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Ode;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Ode")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Ode, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut body__ = None;
                let mut time__ = None;
                let mut state__ = None;
                let mut parameters__ = None;
                let mut output__ = None;
                let mut solver__ = None;
                let mut rtol__ = None;
                let mut atol__ = None;
                let mut max_steps__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Body => {
                            if body__.is_some() {
                                return Err(serde::de::Error::duplicate_field("body"));
                            }
                            body__ = map_.next_value()?;
                        }
                        GeneratedField::Time => {
                            if time__.is_some() {
                                return Err(serde::de::Error::duplicate_field("time"));
                            }
                            time__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::State => {
                            if state__.is_some() {
                                return Err(serde::de::Error::duplicate_field("state"));
                            }
                            state__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Parameters => {
                            if parameters__.is_some() {
                                return Err(serde::de::Error::duplicate_field("parameters"));
                            }
                            parameters__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Output => {
                            if output__.is_some() {
                                return Err(serde::de::Error::duplicate_field("output"));
                            }
                            output__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Solver => {
                            if solver__.is_some() {
                                return Err(serde::de::Error::duplicate_field("solver"));
                            }
                            solver__ = Some(map_.next_value::<OdeSolver>()? as i32);
                        }
                        GeneratedField::Rtol => {
                            if rtol__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rtol"));
                            }
                            rtol__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Atol => {
                            if atol__.is_some() {
                                return Err(serde::de::Error::duplicate_field("atol"));
                            }
                            atol__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::MaxSteps => {
                            if max_steps__.is_some() {
                                return Err(serde::de::Error::duplicate_field("maxSteps"));
                            }
                            max_steps__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Ode {
                    body: body__,
                    time: time__.unwrap_or_default(),
                    state: state__.unwrap_or_default(),
                    parameters: parameters__.unwrap_or_default(),
                    output: output__.unwrap_or_default(),
                    solver: solver__.unwrap_or_default(),
                    rtol: rtol__.unwrap_or_default(),
                    atol: atol__.unwrap_or_default(),
                    max_steps: max_steps__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Ode", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for OdeSolver {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Unspecified => "ODE_SOLVER_UNSPECIFIED",
            Self::Rk45 => "ODE_SOLVER_RK45",
            Self::Stiff => "ODE_SOLVER_STIFF",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for OdeSolver {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "ODE_SOLVER_UNSPECIFIED",
            "ODE_SOLVER_RK45",
            "ODE_SOLVER_STIFF",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = OdeSolver;

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
                    "ODE_SOLVER_UNSPECIFIED" => Ok(OdeSolver::Unspecified),
                    "ODE_SOLVER_RK45" => Ok(OdeSolver::Rk45),
                    "ODE_SOLVER_STIFF" => Ok(OdeSolver::Stiff),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
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
impl serde::Serialize for Scan {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.body.is_some() {
            len += 1;
        }
        if self.length != 0 {
            len += 1;
        }
        if self.carry != 0 {
            len += 1;
        }
        if !self.xs.is_empty() {
            len += 1;
        }
        if !self.constants.is_empty() {
            len += 1;
        }
        if self.output != 0 {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.Scan", len)?;
        if let Some(v) = self.body.as_ref() {
            struct_ser.serialize_field("body", v)?;
        }
        if self.length != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("length", ToString::to_string(&self.length).as_str())?;
        }
        if self.carry != 0 {
            struct_ser.serialize_field("carry", &self.carry)?;
        }
        if !self.xs.is_empty() {
            struct_ser.serialize_field("xs", &self.xs)?;
        }
        if !self.constants.is_empty() {
            struct_ser.serialize_field("constants", &self.constants)?;
        }
        if self.output != 0 {
            struct_ser.serialize_field("output", &self.output)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Scan {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "body",
            "length",
            "carry",
            "xs",
            "constants",
            "output",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Body,
            Length,
            Carry,
            Xs,
            Constants,
            Output,
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
                            "body" => Ok(GeneratedField::Body),
                            "length" => Ok(GeneratedField::Length),
                            "carry" => Ok(GeneratedField::Carry),
                            "xs" => Ok(GeneratedField::Xs),
                            "constants" => Ok(GeneratedField::Constants),
                            "output" => Ok(GeneratedField::Output),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = Scan;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.Scan")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Scan, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut body__ = None;
                let mut length__ = None;
                let mut carry__ = None;
                let mut xs__ = None;
                let mut constants__ = None;
                let mut output__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Body => {
                            if body__.is_some() {
                                return Err(serde::de::Error::duplicate_field("body"));
                            }
                            body__ = map_.next_value()?;
                        }
                        GeneratedField::Length => {
                            if length__.is_some() {
                                return Err(serde::de::Error::duplicate_field("length"));
                            }
                            length__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Carry => {
                            if carry__.is_some() {
                                return Err(serde::de::Error::duplicate_field("carry"));
                            }
                            carry__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Xs => {
                            if xs__.is_some() {
                                return Err(serde::de::Error::duplicate_field("xs"));
                            }
                            xs__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Constants => {
                            if constants__.is_some() {
                                return Err(serde::de::Error::duplicate_field("constants"));
                            }
                            constants__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::Output => {
                            if output__.is_some() {
                                return Err(serde::de::Error::duplicate_field("output"));
                            }
                            output__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                    }
                }
                Ok(Scan {
                    body: body__,
                    length: length__.unwrap_or_default(),
                    carry: carry__.unwrap_or_default(),
                    xs: xs__.unwrap_or_default(),
                    constants: constants__.unwrap_or_default(),
                    output: output__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.Scan", FIELDS, GeneratedVisitor)
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
impl serde::Serialize for SolveTriangular {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.lower {
            len += 1;
        }
        if self.transpose {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.SolveTriangular", len)?;
        if self.lower {
            struct_ser.serialize_field("lower", &self.lower)?;
        }
        if self.transpose {
            struct_ser.serialize_field("transpose", &self.transpose)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SolveTriangular {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "lower",
            "transpose",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Lower,
            Transpose,
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
                            "lower" => Ok(GeneratedField::Lower),
                            "transpose" => Ok(GeneratedField::Transpose),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = SolveTriangular;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.SolveTriangular")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SolveTriangular, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut lower__ = None;
                let mut transpose__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Lower => {
                            if lower__.is_some() {
                                return Err(serde::de::Error::duplicate_field("lower"));
                            }
                            lower__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Transpose => {
                            if transpose__.is_some() {
                                return Err(serde::de::Error::duplicate_field("transpose"));
                            }
                            transpose__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(SolveTriangular {
                    lower: lower__.unwrap_or_default(),
                    transpose: transpose__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.SolveTriangular", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for SparseMatrix {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.rows != 0 {
            len += 1;
        }
        if self.cols != 0 {
            len += 1;
        }
        if !self.row_indices.is_empty() {
            len += 1;
        }
        if !self.col_indices.is_empty() {
            len += 1;
        }
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("keanu.SparseMatrix", len)?;
        if self.rows != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("rows", ToString::to_string(&self.rows).as_str())?;
        }
        if self.cols != 0 {
            #[allow(clippy::needless_borrow)]
            #[allow(clippy::needless_borrows_for_generic_args)]
            struct_ser.serialize_field("cols", ToString::to_string(&self.cols).as_str())?;
        }
        if !self.row_indices.is_empty() {
            struct_ser.serialize_field("rowIndices", &self.row_indices.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.col_indices.is_empty() {
            struct_ser.serialize_field("colIndices", &self.col_indices.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for SparseMatrix {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "rows",
            "cols",
            "row_indices",
            "rowIndices",
            "col_indices",
            "colIndices",
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Rows,
            Cols,
            RowIndices,
            ColIndices,
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
                            "rows" => Ok(GeneratedField::Rows),
                            "cols" => Ok(GeneratedField::Cols),
                            "rowIndices" | "row_indices" => Ok(GeneratedField::RowIndices),
                            "colIndices" | "col_indices" => Ok(GeneratedField::ColIndices),
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
            type Value = SparseMatrix;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct keanu.SparseMatrix")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<SparseMatrix, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut rows__ = None;
                let mut cols__ = None;
                let mut row_indices__ = None;
                let mut col_indices__ = None;
                let mut values__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Rows => {
                            if rows__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rows"));
                            }
                            rows__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::Cols => {
                            if cols__.is_some() {
                                return Err(serde::de::Error::duplicate_field("cols"));
                            }
                            cols__ = 
                                Some(map_.next_value::<::pbjson::private::NumberDeserialize<_>>()?.0)
                            ;
                        }
                        GeneratedField::RowIndices => {
                            if row_indices__.is_some() {
                                return Err(serde::de::Error::duplicate_field("rowIndices"));
                            }
                            row_indices__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
                        GeneratedField::ColIndices => {
                            if col_indices__.is_some() {
                                return Err(serde::de::Error::duplicate_field("colIndices"));
                            }
                            col_indices__ = 
                                Some(map_.next_value::<Vec<::pbjson::private::NumberDeserialize<_>>>()?
                                    .into_iter().map(|x| x.0).collect())
                            ;
                        }
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
                Ok(SparseMatrix {
                    rows: rows__.unwrap_or_default(),
                    cols: cols__.unwrap_or_default(),
                    row_indices: row_indices__.unwrap_or_default(),
                    col_indices: col_indices__.unwrap_or_default(),
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("keanu.SparseMatrix", FIELDS, GeneratedVisitor)
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
            Self::LambertW0 => "UNARY_OP_LAMBERT_W0",
            Self::LambertWm1 => "UNARY_OP_LAMBERT_WM1",
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
            "UNARY_OP_LAMBERT_W0",
            "UNARY_OP_LAMBERT_WM1",
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
                    "UNARY_OP_LAMBERT_W0" => Ok(UnaryOp::LambertW0),
                    "UNARY_OP_LAMBERT_WM1" => Ok(UnaryOp::LambertWm1),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
