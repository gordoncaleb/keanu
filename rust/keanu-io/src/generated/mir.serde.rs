impl serde::Serialize for BooleanTensor {
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
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.BooleanTensor", len)?;
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for BooleanTensor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shape",
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Shape,
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
                            "shape" => Ok(GeneratedField::Shape),
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
            type Value = BooleanTensor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.BooleanTensor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<BooleanTensor, V::Error>
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
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(BooleanTensor {
                    shape: shape__.unwrap_or_default(),
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.BooleanTensor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for DataType {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let variant = match self {
            Self::Double => "DOUBLE",
            Self::Integer => "INTEGER",
            Self::Boolean => "BOOLEAN",
            Self::DoubleCollection => "DOUBLE_COLLECTION",
            Self::IntegerCollection => "INTEGER_COLLECTION",
            Self::BooleanCollection => "BOOLEAN_COLLECTION",
        };
        serializer.serialize_str(variant)
    }
}
impl<'de> serde::Deserialize<'de> for DataType {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "DOUBLE",
            "INTEGER",
            "BOOLEAN",
            "DOUBLE_COLLECTION",
            "INTEGER_COLLECTION",
            "BOOLEAN_COLLECTION",
        ];

        struct GeneratedVisitor;

        impl serde::de::Visitor<'_> for GeneratedVisitor {
            type Value = DataType;

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
                    "DOUBLE" => Ok(DataType::Double),
                    "INTEGER" => Ok(DataType::Integer),
                    "BOOLEAN" => Ok(DataType::Boolean),
                    "DOUBLE_COLLECTION" => Ok(DataType::DoubleCollection),
                    "INTEGER_COLLECTION" => Ok(DataType::IntegerCollection),
                    "BOOLEAN_COLLECTION" => Ok(DataType::BooleanCollection),
                    _ => Err(serde::de::Error::unknown_variant(value, FIELDS)),
                }
            }
        }
        deserializer.deserialize_any(GeneratedVisitor)
    }
}
impl serde::Serialize for DoubleTensor {
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
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.DoubleTensor", len)?;
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for DoubleTensor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shape",
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Shape,
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
                            "shape" => Ok(GeneratedField::Shape),
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
            type Value = DoubleTensor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.DoubleTensor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<DoubleTensor, V::Error>
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
                Ok(DoubleTensor {
                    shape: shape__.unwrap_or_default(),
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.DoubleTensor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for GenericTensor {
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
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.GenericTensor", len)?;
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for GenericTensor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shape",
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Shape,
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
                            "shape" => Ok(GeneratedField::Shape),
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
            type Value = GenericTensor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.GenericTensor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<GenericTensor, V::Error>
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
                        GeneratedField::Values => {
                            if values__.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(GenericTensor {
                    shape: shape__.unwrap_or_default(),
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.GenericTensor", FIELDS, GeneratedVisitor)
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
        if !self.vertices.is_empty() {
            len += 1;
        }
        if !self.default_state.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.Graph", len)?;
        if !self.vertices.is_empty() {
            struct_ser.serialize_field("vertices", &self.vertices)?;
        }
        if !self.default_state.is_empty() {
            struct_ser.serialize_field("defaultState", &self.default_state)?;
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
            "vertices",
            "default_state",
            "defaultState",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Vertices,
            DefaultState,
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
                            "vertices" => Ok(GeneratedField::Vertices),
                            "defaultState" | "default_state" => Ok(GeneratedField::DefaultState),
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
                formatter.write_str("struct mir.Graph")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Graph, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut vertices__ = None;
                let mut default_state__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Vertices => {
                            if vertices__.is_some() {
                                return Err(serde::de::Error::duplicate_field("vertices"));
                            }
                            vertices__ = Some(map_.next_value()?);
                        }
                        GeneratedField::DefaultState => {
                            if default_state__.is_some() {
                                return Err(serde::de::Error::duplicate_field("defaultState"));
                            }
                            default_state__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(Graph {
                    vertices: vertices__.unwrap_or_default(),
                    default_state: default_state__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.Graph", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for IntArray {
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
        let mut struct_ser = serializer.serialize_struct("mir.IntArray", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for IntArray {
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
            type Value = IntArray;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.IntArray")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<IntArray, V::Error>
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
                Ok(IntArray {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.IntArray", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for IntegerTensor {
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
        if !self.values.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.IntegerTensor", len)?;
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for IntegerTensor {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "shape",
            "values",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Shape,
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
                            "shape" => Ok(GeneratedField::Shape),
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
            type Value = IntegerTensor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.IntegerTensor")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<IntegerTensor, V::Error>
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
                Ok(IntegerTensor {
                    shape: shape__.unwrap_or_default(),
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.IntegerTensor", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for LongArray {
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
        let mut struct_ser = serializer.serialize_struct("mir.LongArray", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for LongArray {
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
            type Value = LongArray;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.LongArray")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<LongArray, V::Error>
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
                Ok(LongArray {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.LongArray", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Metadata {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.key.is_empty() {
            len += 1;
        }
        if !self.value.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.Metadata", len)?;
        if !self.key.is_empty() {
            struct_ser.serialize_field("key", &self.key)?;
        }
        if !self.value.is_empty() {
            struct_ser.serialize_field("value", &self.value)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Metadata {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "key",
            "value",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Key,
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
                            "key" => Ok(GeneratedField::Key),
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
            type Value = Metadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.Metadata")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Metadata, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut key__ = None;
                let mut value__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Key => {
                            if key__.is_some() {
                                return Err(serde::de::Error::duplicate_field("key"));
                            }
                            key__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Value => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("value"));
                            }
                            value__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(Metadata {
                    key: key__.unwrap_or_default(),
                    value: value__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.Metadata", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ModelMetadata {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.metadata_info.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.ModelMetadata", len)?;
        if !self.metadata_info.is_empty() {
            struct_ser.serialize_field("metadataInfo", &self.metadata_info)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ModelMetadata {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "metadataInfo",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            MetadataInfo,
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
                            "metadataInfo" => Ok(GeneratedField::MetadataInfo),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = ModelMetadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.ModelMetadata")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ModelMetadata, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut metadata_info__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::MetadataInfo => {
                            if metadata_info__.is_some() {
                                return Err(serde::de::Error::duplicate_field("metadataInfo"));
                            }
                            metadata_info__ = Some(
                                map_.next_value::<std::collections::HashMap<_, _>>()?
                            );
                        }
                    }
                }
                Ok(ModelMetadata {
                    metadata_info: metadata_info__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.ModelMetadata", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for NamedParam {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.name.is_empty() {
            len += 1;
        }
        if self.param.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.NamedParam", len)?;
        if !self.name.is_empty() {
            struct_ser.serialize_field("name", &self.name)?;
        }
        if let Some(v) = self.param.as_ref() {
            match v {
                named_param::Param::DoubleTensorParam(v) => {
                    struct_ser.serialize_field("doubleTensorParam", v)?;
                }
                named_param::Param::IntTensorParam(v) => {
                    struct_ser.serialize_field("intTensorParam", v)?;
                }
                named_param::Param::BoolTensorParam(v) => {
                    struct_ser.serialize_field("boolTensorParam", v)?;
                }
                named_param::Param::ParentVertex(v) => {
                    struct_ser.serialize_field("parentVertex", v)?;
                }
                named_param::Param::DoubleParam(v) => {
                    struct_ser.serialize_field("doubleParam", v)?;
                }
                named_param::Param::IntParam(v) => {
                    struct_ser.serialize_field("intParam", v)?;
                }
                named_param::Param::LongParam(v) => {
                    #[allow(clippy::needless_borrow)]
                    #[allow(clippy::needless_borrows_for_generic_args)]
                    struct_ser.serialize_field("longParam", ToString::to_string(&v).as_str())?;
                }
                named_param::Param::StringParam(v) => {
                    struct_ser.serialize_field("stringParam", v)?;
                }
                named_param::Param::LongArrayParam(v) => {
                    struct_ser.serialize_field("longArrayParam", v)?;
                }
                named_param::Param::VertexArrayParam(v) => {
                    struct_ser.serialize_field("vertexArrayParam", v)?;
                }
                named_param::Param::IntArrayParam(v) => {
                    struct_ser.serialize_field("intArrayParam", v)?;
                }
                named_param::Param::BoolParam(v) => {
                    struct_ser.serialize_field("boolParam", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for NamedParam {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "name",
            "double_tensor_param",
            "doubleTensorParam",
            "int_tensor_param",
            "intTensorParam",
            "bool_tensor_param",
            "boolTensorParam",
            "parent_vertex",
            "parentVertex",
            "double_param",
            "doubleParam",
            "int_param",
            "intParam",
            "long_param",
            "longParam",
            "string_param",
            "stringParam",
            "long_array_param",
            "longArrayParam",
            "vertex_array_param",
            "vertexArrayParam",
            "int_array_param",
            "intArrayParam",
            "bool_param",
            "boolParam",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Name,
            DoubleTensorParam,
            IntTensorParam,
            BoolTensorParam,
            ParentVertex,
            DoubleParam,
            IntParam,
            LongParam,
            StringParam,
            LongArrayParam,
            VertexArrayParam,
            IntArrayParam,
            BoolParam,
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
                            "name" => Ok(GeneratedField::Name),
                            "doubleTensorParam" | "double_tensor_param" => Ok(GeneratedField::DoubleTensorParam),
                            "intTensorParam" | "int_tensor_param" => Ok(GeneratedField::IntTensorParam),
                            "boolTensorParam" | "bool_tensor_param" => Ok(GeneratedField::BoolTensorParam),
                            "parentVertex" | "parent_vertex" => Ok(GeneratedField::ParentVertex),
                            "doubleParam" | "double_param" => Ok(GeneratedField::DoubleParam),
                            "intParam" | "int_param" => Ok(GeneratedField::IntParam),
                            "longParam" | "long_param" => Ok(GeneratedField::LongParam),
                            "stringParam" | "string_param" => Ok(GeneratedField::StringParam),
                            "longArrayParam" | "long_array_param" => Ok(GeneratedField::LongArrayParam),
                            "vertexArrayParam" | "vertex_array_param" => Ok(GeneratedField::VertexArrayParam),
                            "intArrayParam" | "int_array_param" => Ok(GeneratedField::IntArrayParam),
                            "boolParam" | "bool_param" => Ok(GeneratedField::BoolParam),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = NamedParam;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.NamedParam")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<NamedParam, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut name__ = None;
                let mut param__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Name => {
                            if name__.is_some() {
                                return Err(serde::de::Error::duplicate_field("name"));
                            }
                            name__ = Some(map_.next_value()?);
                        }
                        GeneratedField::DoubleTensorParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("doubleTensorParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::DoubleTensorParam)
;
                        }
                        GeneratedField::IntTensorParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("intTensorParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::IntTensorParam)
;
                        }
                        GeneratedField::BoolTensorParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("boolTensorParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::BoolTensorParam)
;
                        }
                        GeneratedField::ParentVertex => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("parentVertex"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::ParentVertex)
;
                        }
                        GeneratedField::DoubleParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("doubleParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<::pbjson::private::NumberDeserialize<_>>>()?.map(|x| named_param::Param::DoubleParam(x.0));
                        }
                        GeneratedField::IntParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("intParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<::pbjson::private::NumberDeserialize<_>>>()?.map(|x| named_param::Param::IntParam(x.0));
                        }
                        GeneratedField::LongParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("longParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<::pbjson::private::NumberDeserialize<_>>>()?.map(|x| named_param::Param::LongParam(x.0));
                        }
                        GeneratedField::StringParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("stringParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::StringParam);
                        }
                        GeneratedField::LongArrayParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("longArrayParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::LongArrayParam)
;
                        }
                        GeneratedField::VertexArrayParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("vertexArrayParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::VertexArrayParam)
;
                        }
                        GeneratedField::IntArrayParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("intArrayParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::IntArrayParam)
;
                        }
                        GeneratedField::BoolParam => {
                            if param__.is_some() {
                                return Err(serde::de::Error::duplicate_field("boolParam"));
                            }
                            param__ = map_.next_value::<::std::option::Option<_>>()?.map(named_param::Param::BoolParam);
                        }
                    }
                }
                Ok(NamedParam {
                    name: name__.unwrap_or_default(),
                    param: param__,
                })
            }
        }
        deserializer.deserialize_struct("mir.NamedParam", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for ProtoModel {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.graph.is_some() {
            len += 1;
        }
        if self.metadata.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.ProtoModel", len)?;
        if let Some(v) = self.graph.as_ref() {
            struct_ser.serialize_field("graph", v)?;
        }
        if let Some(v) = self.metadata.as_ref() {
            struct_ser.serialize_field("metadata", v)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for ProtoModel {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "graph",
            "metadata",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
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
            type Value = ProtoModel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.ProtoModel")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<ProtoModel, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut graph__ = None;
                let mut metadata__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
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
                            metadata__ = map_.next_value()?;
                        }
                    }
                }
                Ok(ProtoModel {
                    graph: graph__,
                    metadata: metadata__,
                })
            }
        }
        deserializer.deserialize_struct("mir.ProtoModel", FIELDS, GeneratedVisitor)
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
        if self.value.is_some() {
            len += 1;
        }
        if !self.vertex_label.is_empty() {
            len += 1;
        }
        if self.id.is_some() {
            len += 1;
        }
        if self.is_observed {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.StoredValue", len)?;
        if let Some(v) = self.value.as_ref() {
            struct_ser.serialize_field("value", v)?;
        }
        if !self.vertex_label.is_empty() {
            struct_ser.serialize_field("vertexLabel", &self.vertex_label)?;
        }
        if let Some(v) = self.id.as_ref() {
            struct_ser.serialize_field("id", v)?;
        }
        if self.is_observed {
            struct_ser.serialize_field("isObserved", &self.is_observed)?;
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
            "value",
            "vertex_label",
            "vertexLabel",
            "id",
            "isObserved",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Value,
            VertexLabel,
            Id,
            IsObserved,
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
                            "vertexLabel" | "vertex_label" => Ok(GeneratedField::VertexLabel),
                            "id" => Ok(GeneratedField::Id),
                            "isObserved" => Ok(GeneratedField::IsObserved),
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
                formatter.write_str("struct mir.StoredValue")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<StoredValue, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut value__ = None;
                let mut vertex_label__ = None;
                let mut id__ = None;
                let mut is_observed__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Value => {
                            if value__.is_some() {
                                return Err(serde::de::Error::duplicate_field("value"));
                            }
                            value__ = map_.next_value()?;
                        }
                        GeneratedField::VertexLabel => {
                            if vertex_label__.is_some() {
                                return Err(serde::de::Error::duplicate_field("vertexLabel"));
                            }
                            vertex_label__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Id => {
                            if id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id__ = map_.next_value()?;
                        }
                        GeneratedField::IsObserved => {
                            if is_observed__.is_some() {
                                return Err(serde::de::Error::duplicate_field("isObserved"));
                            }
                            is_observed__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(StoredValue {
                    value: value__,
                    vertex_label: vertex_label__.unwrap_or_default(),
                    id: id__,
                    is_observed: is_observed__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.StoredValue", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for Vertex {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.id.is_some() {
            len += 1;
        }
        if !self.label.is_empty() {
            len += 1;
        }
        if !self.vertex_type.is_empty() {
            len += 1;
        }
        if !self.parameters.is_empty() {
            len += 1;
        }
        if !self.shape.is_empty() {
            len += 1;
        }
        if self.data_type != 0 {
            len += 1;
        }
        if !self.metadata.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.Vertex", len)?;
        if let Some(v) = self.id.as_ref() {
            struct_ser.serialize_field("id", v)?;
        }
        if !self.label.is_empty() {
            struct_ser.serialize_field("label", &self.label)?;
        }
        if !self.vertex_type.is_empty() {
            struct_ser.serialize_field("vertexType", &self.vertex_type)?;
        }
        if !self.parameters.is_empty() {
            struct_ser.serialize_field("parameters", &self.parameters)?;
        }
        if !self.shape.is_empty() {
            struct_ser.serialize_field("shape", &self.shape.iter().map(ToString::to_string).collect::<Vec<_>>())?;
        }
        if self.data_type != 0 {
            let v = DataType::try_from(self.data_type)
                .map_err(|_| serde::ser::Error::custom(format!("Invalid variant {}", self.data_type)))?;
            struct_ser.serialize_field("dataType", &v)?;
        }
        if !self.metadata.is_empty() {
            struct_ser.serialize_field("metadata", &self.metadata)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for Vertex {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "id",
            "label",
            "vertexType",
            "parameters",
            "shape",
            "data_type",
            "dataType",
            "metadata",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Id,
            Label,
            VertexType,
            Parameters,
            Shape,
            DataType,
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
                            "id" => Ok(GeneratedField::Id),
                            "label" => Ok(GeneratedField::Label),
                            "vertexType" => Ok(GeneratedField::VertexType),
                            "parameters" => Ok(GeneratedField::Parameters),
                            "shape" => Ok(GeneratedField::Shape),
                            "dataType" | "data_type" => Ok(GeneratedField::DataType),
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
            type Value = Vertex;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.Vertex")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<Vertex, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut id__ = None;
                let mut label__ = None;
                let mut vertex_type__ = None;
                let mut parameters__ = None;
                let mut shape__ = None;
                let mut data_type__ = None;
                let mut metadata__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Id => {
                            if id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id__ = map_.next_value()?;
                        }
                        GeneratedField::Label => {
                            if label__.is_some() {
                                return Err(serde::de::Error::duplicate_field("label"));
                            }
                            label__ = Some(map_.next_value()?);
                        }
                        GeneratedField::VertexType => {
                            if vertex_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("vertexType"));
                            }
                            vertex_type__ = Some(map_.next_value()?);
                        }
                        GeneratedField::Parameters => {
                            if parameters__.is_some() {
                                return Err(serde::de::Error::duplicate_field("parameters"));
                            }
                            parameters__ = Some(map_.next_value()?);
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
                        GeneratedField::DataType => {
                            if data_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("dataType"));
                            }
                            data_type__ = Some(map_.next_value::<DataType>()? as i32);
                        }
                        GeneratedField::Metadata => {
                            if metadata__.is_some() {
                                return Err(serde::de::Error::duplicate_field("metadata"));
                            }
                            metadata__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(Vertex {
                    id: id__,
                    label: label__.unwrap_or_default(),
                    vertex_type: vertex_type__.unwrap_or_default(),
                    parameters: parameters__.unwrap_or_default(),
                    shape: shape__.unwrap_or_default(),
                    data_type: data_type__.unwrap_or_default(),
                    metadata: metadata__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.Vertex", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for VertexArray {
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
        let mut struct_ser = serializer.serialize_struct("mir.VertexArray", len)?;
        if !self.values.is_empty() {
            struct_ser.serialize_field("values", &self.values)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for VertexArray {
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
            type Value = VertexArray;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.VertexArray")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<VertexArray, V::Error>
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
                Ok(VertexArray {
                    values: values__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.VertexArray", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for VertexId {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if !self.id.is_empty() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.VertexID", len)?;
        if !self.id.is_empty() {
            struct_ser.serialize_field("id", &self.id)?;
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for VertexId {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "id",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            Id,
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
                            "id" => Ok(GeneratedField::Id),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = VertexId;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.VertexID")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<VertexId, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut id__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::Id => {
                            if id__.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id__ = Some(map_.next_value()?);
                        }
                    }
                }
                Ok(VertexId {
                    id: id__.unwrap_or_default(),
                })
            }
        }
        deserializer.deserialize_struct("mir.VertexID", FIELDS, GeneratedVisitor)
    }
}
impl serde::Serialize for VertexValue {
    #[allow(deprecated)]
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut len = 0;
        if self.value_type.is_some() {
            len += 1;
        }
        let mut struct_ser = serializer.serialize_struct("mir.VertexValue", len)?;
        if let Some(v) = self.value_type.as_ref() {
            match v {
                vertex_value::ValueType::DoubleVal(v) => {
                    struct_ser.serialize_field("doubleVal", v)?;
                }
                vertex_value::ValueType::IntVal(v) => {
                    struct_ser.serialize_field("intVal", v)?;
                }
                vertex_value::ValueType::BoolVal(v) => {
                    struct_ser.serialize_field("boolVal", v)?;
                }
                vertex_value::ValueType::GenericVal(v) => {
                    struct_ser.serialize_field("genericVal", v)?;
                }
            }
        }
        struct_ser.end()
    }
}
impl<'de> serde::Deserialize<'de> for VertexValue {
    #[allow(deprecated)]
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &[
            "double_val",
            "doubleVal",
            "int_val",
            "intVal",
            "bool_val",
            "boolVal",
            "generic_val",
            "genericVal",
        ];

        #[allow(clippy::enum_variant_names)]
        enum GeneratedField {
            DoubleVal,
            IntVal,
            BoolVal,
            GenericVal,
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
                            "doubleVal" | "double_val" => Ok(GeneratedField::DoubleVal),
                            "intVal" | "int_val" => Ok(GeneratedField::IntVal),
                            "boolVal" | "bool_val" => Ok(GeneratedField::BoolVal),
                            "genericVal" | "generic_val" => Ok(GeneratedField::GenericVal),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(GeneratedVisitor)
            }
        }
        struct GeneratedVisitor;
        impl<'de> serde::de::Visitor<'de> for GeneratedVisitor {
            type Value = VertexValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("struct mir.VertexValue")
            }

            fn visit_map<V>(self, mut map_: V) -> std::result::Result<VertexValue, V::Error>
                where
                    V: serde::de::MapAccess<'de>,
            {
                let mut value_type__ = None;
                while let Some(k) = map_.next_key()? {
                    match k {
                        GeneratedField::DoubleVal => {
                            if value_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("doubleVal"));
                            }
                            value_type__ = map_.next_value::<::std::option::Option<_>>()?.map(vertex_value::ValueType::DoubleVal)
;
                        }
                        GeneratedField::IntVal => {
                            if value_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("intVal"));
                            }
                            value_type__ = map_.next_value::<::std::option::Option<_>>()?.map(vertex_value::ValueType::IntVal)
;
                        }
                        GeneratedField::BoolVal => {
                            if value_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("boolVal"));
                            }
                            value_type__ = map_.next_value::<::std::option::Option<_>>()?.map(vertex_value::ValueType::BoolVal)
;
                        }
                        GeneratedField::GenericVal => {
                            if value_type__.is_some() {
                                return Err(serde::de::Error::duplicate_field("genericVal"));
                            }
                            value_type__ = map_.next_value::<::std::option::Option<_>>()?.map(vertex_value::ValueType::GenericVal)
;
                        }
                    }
                }
                Ok(VertexValue {
                    value_type: value_type__,
                })
            }
        }
        deserializer.deserialize_struct("mir.VertexValue", FIELDS, GeneratedVisitor)
    }
}
