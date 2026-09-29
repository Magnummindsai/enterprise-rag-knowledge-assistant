---
doc_id: PAC
title: Password & Access Control Standard
owner: Information Security
version: 2.4
effective_date: 2026-03-01
---

# Password & Access Control Standard

> Synthetic document for demonstration purposes. Northwind Dynamics is a fictional company.

## 1. Purpose
This standard defines minimum requirements for passwords, multi-factor authentication (MFA) and access to Northwind Dynamics systems. It supports the IT Acceptable Use & Security Policy (ITS).

## 2. Passwords

### 2.1 Password Requirements
User account passwords must be at least 14 characters long. Passphrases of 4 or more unrelated words are encouraged. Passwords must not contain the user's name, username or the company name, and must not appear on the known-breached password list, which is checked automatically.

### 2.2 Password Rotation
Passwords do not expire on a fixed schedule. A password must be changed immediately if there is any suspicion that it has been compromised, or when requested by Information Security.

### 2.3 Password Reuse
Company passwords must not be reused for any personal or third-party service. The last 12 passwords cannot be reused.

### 2.4 Password Manager
Employees must use the company-provided password manager (VaultKey) to store work credentials. Storing passwords in browsers, spreadsheets, notes or sticky notes is prohibited.

### 2.5 Account Lockout
Accounts are locked for 15 minutes after 10 consecutive failed login attempts. Repeated lockouts are investigated by the Security Operations Centre.

## 3. Multi-Factor Authentication

### 3.1 MFA Requirement
MFA is mandatory for all remote access, email, VPN, cloud applications and administrative access.

### 3.2 Approved Methods
Approved MFA methods are the Authenticator app with number matching and FIDO2 hardware security keys. SMS codes are not an approved method except as a temporary fallback for up to 7 days while a replacement device is issued.

### 3.3 Privileged Accounts
Privileged and administrator accounts must use FIDO2 hardware security keys.

## 4. Access Control

### 4.1 Least Privilege
Access is granted on the principle of least privilege: users receive only the access required for their role.

### 4.2 Requesting Access
Access to systems must be requested through the IT Service Desk portal and approved by the system owner and the requester's line manager.

### 4.3 Privileged Access
Administrator access is granted through separate named admin accounts, never through a user's day-to-day account. Privileged access is time-limited to a maximum of 8 hours per elevation via the Privileged Access Management tool.

### 4.4 Access Reviews
System owners must review user access to Confidential and Restricted systems every quarter, and to all other systems every 6 months.

### 4.5 Joiners, Movers and Leavers
When an employee changes role, access no longer required must be removed within 5 working days. When an employee leaves, all access must be disabled by the end of their last working day. For involuntary terminations, access is disabled immediately.

## 5. Shared and Service Accounts

### 5.1 Shared Accounts
Shared accounts are prohibited unless there is no technical alternative and Information Security has granted an exception. Credentials for approved shared accounts must be stored in VaultKey and rotated whenever a member of the group leaves.

### 5.2 Service Accounts
Service account credentials must be stored in the secrets manager, rotated at least every 90 days, and must never be hard-coded in source code.

## 6. Exceptions
Exceptions to this standard must be requested from Information Security, are valid for a maximum of 6 months, and must be documented in the risk register.
