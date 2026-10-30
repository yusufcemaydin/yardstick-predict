export function page(title, body) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>${escape(title)} · Storefront</title>
</head>
<body>
  <nav><a href="/">Storefront</a></nav>
  <main>
    <h1>${escape(title)}</h1>
    ${body}
  </main>
</body>
</html>`;
}

export function productList(products) {
  const items = products
    .map((product) => `<li><a href="/products/${product.id}">${escape(product.name)}</a> $${product.price}</li>`)
    .join("\n");
  return `<ul class="products">\n${items}\n</ul>`;
}

export function productDetail(product) {
  return `<p>${escape(product.description)}</p>\n<p class="price">$${product.price}</p>`;
}

export function escape(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
