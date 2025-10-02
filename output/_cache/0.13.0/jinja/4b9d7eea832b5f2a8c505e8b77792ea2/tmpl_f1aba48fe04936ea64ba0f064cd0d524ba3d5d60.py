from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'screens/search/legend.jinja'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_view_object = resolve('view_object')
    pass
    yield '<sdoc-main-legend\n  data-testid="search-main-legend"\n>\n    <p>The search query syntax is inspired by Python and is based on a fixed grammar.</p>\n    <p>Important rules:</p>\n    <ul>\n      <li>\n        Every query component shall start with <code>node.</code>.\n      </li>\n      <li>\n        <code>and</code> and <code>or</code> expressions must be grouped using round brackets.\n      </li>\n      <li>\n        Only double quotes are accepted for strings.\n      </li>\n    </ul>\n    <p>Examples:</p>\n    <div class="sdoc-table_key_value">\n      <a\n        class="sdoc-table_key_value-key"\n        data-testid="node.is_requirement"\n        href="'
    yield escape(context.call(environment.getattr((undefined(name='view_object') if l_0_view_object is missing else l_0_view_object), 'render_url'), 'search?q=node.is_requirement'))
    yield '"\n      >\n        <code>node.is_requirement</code>\n      </a>\n      <div class="sdoc-table_key_value-value">Find all requirements.</div>\n      <a\n        class="sdoc-table_key_value-key"\n        data-testid="node.is_section"\n        href="'
    yield escape(context.call(environment.getattr((undefined(name='view_object') if l_0_view_object is missing else l_0_view_object), 'render_url'), 'search?q=node.is_section'))
    yield '"\n      >\n        <code>node.is_section</code>\n      </a>\n      <div class="sdoc-table_key_value-value">Find all sections.</div>\n      <a\n        class="sdoc-table_key_value-key"\n        href="'
    yield escape(context.call(environment.getattr((undefined(name='view_object') if l_0_view_object is missing else l_0_view_object), 'render_url'), 'search?q=(node.is_requirement and "System" in node["TITLE"])'))
    yield '"\n        data-testid="node_is_title_system"\n      >\n        <code>\n        (node.is_requirement and "System" in node["TITLE"])</code>\n      </a>\n      <div class="sdoc-table_key_value-value">Find all requirements with a TITLE that equals to "System".</div>\n      <a class="sdoc-table_key_value-key" href="'
    yield escape(context.call(environment.getattr((undefined(name='view_object') if l_0_view_object is missing else l_0_view_object), 'render_url'), 'search?q=(node.is_requirement and node.is_root)'))
    yield '"><code>(node.is_requirement and node.is_root)</code></a>\n      <div class="sdoc-table_key_value-value">Find all root requirements.</div>\n      <a class="sdoc-table_key_value-key" href="'
    yield escape(context.call(environment.getattr((undefined(name='view_object') if l_0_view_object is missing else l_0_view_object), 'render_url'), 'search?q=(node.is_requirement and node.has_parent_requirements)'))
    yield '"><code>(node.is_requirement and node.has_parent_requirements)</code></a>\n      <div class="sdoc-table_key_value-value">Find all requirements which have parent requirements.</div>\n    </div>\n    <p>See the User Guide\'s section "Query engine" for more details.</p>\n</sdoc-main-legend>'

blocks = {}
debug_info = '22=13&30=15&37=17&44=19&46=21'