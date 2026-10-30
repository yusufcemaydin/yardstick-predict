import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";

function account(email, password) {
  const salt = randomBytes(16);
  return { email, salt, hash: scryptSync(password, salt, 32) };
}

const accounts = [account("demo@example.com", "correct horse battery staple")];

export function verifyUser(email, password) {
  const found = accounts.find((user) => user.email === String(email).trim().toLowerCase());
  if (!found) return null;
  const hash = scryptSync(String(password), found.salt, 32);
  return timingSafeEqual(hash, found.hash) ? { email: found.email } : null;
}
