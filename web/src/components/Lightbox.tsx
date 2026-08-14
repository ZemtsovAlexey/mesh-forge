export default function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div className="lightbox" onClick={onClose} role="presentation">
      <img src={src} alt="" />
    </div>
  );
}
