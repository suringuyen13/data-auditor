type_rules = {
    "employee_id": {
        "type": "string"
    },
    "age": {
        "type": "integer"
    },
    "salary": {
        "type": "float"
    },
    "start_date": {
        "type": "date",
        "format": "%Y-%m-%d"
    },
    "approved": {
        "type": "boolean",
        "allowed_values": [
            "true",
            "false",
            "yes",
            "no",
            "1",
            "0"
        ]
    },
    "end_date": {
        "type": "date",
        "format": "%Y-%m-%d"
    }
}