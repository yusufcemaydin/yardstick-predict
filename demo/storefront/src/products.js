export const products = [
  { id: "desk-lamp", name: "Desk lamp", price: 39, description: "A warm light that clamps to any desk." },
  { id: "notebook", name: "Notebook", price: 12, description: "Dotted pages, a hard cover and a ribbon." },
  { id: "mug", name: "Mug", price: 18, description: "Holds a big coffee and keeps it hot." },
  { id: "backpack", name: "Backpack", price: 85, description: "Fits a laptop and a day's things." },
];

export function findProduct(id) {
  return products.find((product) => product.id === id);
}
