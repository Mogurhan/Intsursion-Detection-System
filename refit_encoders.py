from sklearn.preprocessing import LabelEncoder
import joblib

# Define the string categories as in your app.py
protocol_types = ['icmp', 'tcp', 'udp']
service_types = [
    "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "domain_u",
    "echo", "eco_i", "ecr_i", "efs", "exec", "finger", "ftp", "ftp_data", "gopher", "harvest",
    "hostnames", "http", "http_2784", "http_443", "http_8001", "imap4", "IRC", "iso_tsap", "klogin",
    "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn",
    "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump", "pop_2", "pop_3", "printer", "private",
    "red_i", "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat",
    "telnet", "tftp_u", "tim_i", "time", "urh_i", "urp_i", "uucp", "uucp_path", "vmnet", "whois", "X11",
    "Z39_50"
]
flag_types = [
    "OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"
]

# Fit encoders on string values
protocol_type_encoder = LabelEncoder().fit(protocol_types)
service_encoder = LabelEncoder().fit(service_types)
flag_encoder = LabelEncoder().fit(flag_types)

# Save encoders
joblib.dump(protocol_type_encoder, 'models/protocol_type_encoder.pkl')
joblib.dump(service_encoder, 'models/service_encoder.pkl')
joblib.dump(flag_encoder, 'models/flag_encoder.pkl')
print('Encoders saved to models/.') 