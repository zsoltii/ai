import xml.etree.ElementTree as ET

filename = 'huwiki-latest-pages-articles.xml'

count = 0
redirects = 0
namespace_0 = 0 # Ez jelöli a valódi szócikkeket

# Az iterparse nem tölti be az egészet a memóriába
for event, elem in ET.iterparse(filename, events=('end',)):
    if elem.tag.endswith('page'):
        count += 1

        # Ellenőrizzük a névteret (0 = szócikk)
        ns = elem.find('{http://www.mediawiki.org/xml/export-0.11/}ns')
        is_redirect = elem.find('{http://www.mediawiki.org/xml/export-0.11/}redirect') is not None

        if ns is not None and ns.text == '0':
            if is_redirect:
                redirects += 1
            else:
                namespace_0 += 1

        # Memória felszabadítása
        elem.clear()

        if count % 50000 == 0:
            print(f"Feldolgozva: {count} lap...")

print(f"\nÖsszesen talált lap: {count}")
print(f"Valódi szócikk (ns 0): {namespace_0}")
print(f"Átirányítás (redirect): {redirects}")