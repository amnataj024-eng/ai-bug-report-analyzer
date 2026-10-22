# Bug Report: App crashes when tapping Pay on checkout page

**Date:** 2026-10-22 13:11

**Affected Feature:** Checkout page

**Severity:** Critical
*Reason:* The bug causes the application to crash completely when attempting to complete a payment, blocking the core checkout flow without any available workaround.

## Summary
The app crashes when tapping Pay on the checkout page.

## Possible Root Causes
1. Null pointer exception or missing required fields when accessing payment method details or user billing data during payload construction.
2. Unhandled exception or initialization failure within the integrated third-party payment gateway SDK when launching the payment sheet.
3. Type casting or parsing failure on cart total values or currency formatting prior to dispatching the transaction.
4. Race condition or illegal state exception triggered by multiple rapid taps on the Pay button initiating concurrent payment requests.

## Original Description
App crashes when I tap Pay on the checkout page
